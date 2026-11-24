"""Only source donors and separately materialized target references enter draws.

Every requested donor is retained, including failed finite-search severity
matches. One generated partition supplies every MRI descriptor. Draw identity
does not depend on task scheduling, storage batch, or later target performance.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .provenance import (
    RunWriter,
    content_hash,
    file_hash,
    inside,
    rng,
    subseed,
    verify_run,
)
from .prepare_inputs import require_protocol_lock


def _generate_job(args):
    from .adapters import diffuse
    from .adapters.descriptors import describe, descriptor_definedness
    from .adapters.masks import MaskData

    (
        cfg,
        source,
        target,
        unit,
        patient,
        path,
        expected_hash,
        donors,
        feature_ids,
        start_draw,
        end_draw,
    ) = args
    if file_hash(path) != expected_hash:
        raise ValueError(f"Reference mask hash changed: {unit}")
    with np.load(path, allow_pickle=False) as values:
        reference = values["mask"]
        spacing = tuple(values["spacing"].tolist())
        origin = tuple(values["origin"].tolist())
    modality = cfg["cohorts"][target]["modality"]
    if spacing != (1.0, 1.0, 1.0):
        raise ValueError("Generator requires the audited 1 mm grid")
    seg = reference if modality == "MRI" else (reference > 0).astype(np.uint8) * 3
    probabilities = np.asarray([d["probability"] for d in donors])
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Source donor weights are not normalized")
    rows, attempts, flags = [], [], []
    settings = cfg["resolved_generator"]
    for draw in range(start_draw, end_draw):
        started = time.perf_counter()
        keys = [source, target, patient, unit, draw]
        donor_rng = rng(cfg, "severity_donors", *keys)
        donor = donors[int(donor_rng.choice(len(donors), p=probabilities))]
        requested = float(donor["observed_dice"])
        prep = diffuse.prepare(
            seg,
            rng=rng(cfg, "perturbation_fields", *keys),
            sigma_mm=settings["sigma_vox"],
        )
        amp, achieved, iterations, converged = diffuse.calibrate(
            prep,
            seg > 0,
            requested,
            tol=settings["tolerance"],
            max_iter=settings["max_iter"],
            bracket=tuple(settings["amplitude_bracket"]),
        )
        synthetic = diffuse.apply(prep, amp)
        mask_hash = hashlib.sha256(
            np.ascontiguousarray(synthetic).tobytes()
        ).hexdigest()
        features = describe(MaskData(synthetic, origin, spacing), modality)
        if set(feature_ids) - set(features):
            raise ValueError(
                "Generated extraction does not cover the audited descriptor identities"
            )
        row = {
            "cohort": target,
            "patient_id": patient,
            "unit_id": unit,
            "draw_id": draw,
            "batch": draw // 20,
            "coherent_draw_id": content_hash(keys),
            **{key: float(features[key]) for key in feature_ids},
        }
        rows.append(row)
        flags.append(
            {
                key: row[key]
                for key in ["cohort", "patient_id", "unit_id", "draw_id", "batch"]
            }
            | {
                key: descriptor_definedness(features[key], key, synthetic, modality)
                for key in feature_ids
            }
        )
        attempts.append(
            {
                "source": source,
                "target": target,
                "patient_id": patient,
                "unit_id": unit,
                "draw_id": draw,
                "batch": draw // 20,
                "source_patient_id": donor["patient_id"],
                "source_unit_id": donor["unit_id"],
                "source_alternate_annotation_id": donor["alternate_annotation_id"],
                "source_reference_annotation_id": donor["reference_annotation_id"],
                "donor_probability": donor["probability"],
                "requested_dice": requested,
                "achieved_dice": achieved,
                "absolute_dice_discrepancy": abs(achieved - requested),
                "amplitude": amp,
                "iterations": iterations,
                "converged": converged,
                "failure_reason": "" if converged else "severity_tolerance_not_reached",
                "reference_sha256": expected_hash,
                "synthetic_mask_array_sha256": mask_hash,
                "field_seed": str(subseed(cfg, "perturbation_fields", *keys)),
                "donor_seed": str(subseed(cfg, "severity_donors", *keys)),
                "seconds": time.perf_counter() - started,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(attempts), pd.DataFrame(flags)


def generate_primary(
    workspace,
    prepared_run,
    source,
    target,
    run_id,
    *,
    workers=1,
    refine_from=None,
    convergence_run=None,
):
    workspace, prepared_run = Path(workspace).resolve(), Path(prepared_run).resolve()
    cfg = require_protocol_lock(workspace)
    if {"source": source, "target": target} not in cfg["directions"]:
        raise ValueError("Direction is outside the locked study")
    donor_path = prepared_run / source / "source_severity_donors.csv"
    reference_path = prepared_run / target / "reference_mask_manifest.csv"
    prepared_manifest_path = prepared_run / "run_manifest.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if (
        prepared_manifest["status"] != "complete"
        or prepared_manifest["stage"] != "prepare_inputs"
        or prepared_manifest["protocol_sha256"] != content_hash(cfg)
    ):
        raise ValueError(
            "Prepared inputs must be complete under the same execution protocol"
        )
    for path in [donor_path, reference_path]:
        name = path.relative_to(prepared_run).as_posix()
        if prepared_manifest["outputs"].get(name) != file_hash(path):
            raise ValueError(f"Prepared permitted input missing or changed: {name}")
    donor_data = pd.read_csv(
        donor_path, float_precision="round_trip", keep_default_na=False
    )
    for key in [
        "patient_id",
        "unit_id",
        "alternate_annotation_id",
        "reference_annotation_id",
    ]:
        donor_data[key] = donor_data[key].astype(str)
    refs = pd.read_csv(reference_path, dtype=str, keep_default_na=False)
    refs = refs.loc[refs.cohort.eq(target)].sort_values("unit_id")
    subjects = pd.read_csv(
        workspace / "audit/subjects_metadata.csv", dtype=str, keep_default_na=False
    )
    patient_of = dict(
        subjects.loc[
            subjects.cohort_id.eq(target), ["unit_id", "patient_id"]
        ].to_numpy()
    )
    if refs.unit_id.duplicated().any() or set(refs.unit_id) != set(patient_of):
        raise ValueError(
            "Prepared reference units differ from the audited target cohort"
        )
    if set(donor_data.cohort) != {source}:
        raise ValueError("Severity donors belong to another cohort")
    source_units = set(subjects.loc[subjects.cohort_id.eq(source), "unit_id"])
    if set(donor_data.unit_id) != source_units:
        raise ValueError("Source severity does not cover the audited source units")
    manifest_path = workspace / "audit/audited_descriptor_manifest.csv"
    descriptors = pd.read_csv(manifest_path, keep_default_na=False)
    ids = descriptors.loc[
        descriptors.domain.eq(cfg["cohorts"][target]["modality"]), "feature_id"
    ].tolist()
    start, end = 0, cfg["synthetic"]["draw_count_initial"]
    old_values = old_calibration = old_flags = None
    refinement_inputs = []
    if refine_from is not None:
        refine_from = Path(refine_from).resolve()
        if convergence_run is None:
            raise ValueError(
                "Refinement requires the pre-evaluation convergence artifact"
            )
        convergence_run = Path(convergence_run).resolve()
        for run in [refine_from, convergence_run]:
            if verify_run(run):
                raise ValueError(f"Refinement dependency integrity failed: {run}")
        original = json.loads((refine_from / "run_manifest.json").read_text())
        check = json.loads((convergence_run / "status.json").read_text())
        extra = original["extra"]
        if (
            original["protocol_sha256"] != content_hash(cfg)
            or extra["source"] != source
            or extra["target"] != target
            or extra["draw_count"] != 100
            or not check["refinement_required"]
            or check["draw_count"] != 100
            or check["source"] != source
            or check["target"] != target
        ):
            raise ValueError(
                "Only the declared one-time, precision-triggered 100-to-200 refinement is allowed"
            )
        start, end = 100, cfg["synthetic"]["convergence"]["refined_draw_count"]
        old_values = pd.read_csv(
            refine_from / "draw_features.csv", float_precision="round_trip"
        )
        old_calibration = pd.read_csv(
            refine_from / "calibration_attempts.csv", float_precision="round_trip"
        )
        old_flags = pd.read_csv(
            refine_from / "draw_definedness.csv", keep_default_na=False
        )
        refinement_inputs = [
            refine_from / name
            for name in [
                "run_manifest.json",
                "draw_features.csv",
                "calibration_attempts.csv",
                "draw_definedness.csv",
            ]
        ]
        refinement_inputs += [
            convergence_run / "run_manifest.json",
            convergence_run / "status.json",
        ]
    inputs = [
        workspace / "config/protocol.lock.yaml",
        donor_path,
        reference_path,
        manifest_path,
        workspace / "audit/subjects_metadata.csv",
        prepared_manifest_path,
        workspace / cfg["resolved_metadata"]["patient_split"],
        *refinement_inputs,
    ]
    jobs = []
    for row in refs.itertuples():
        path = inside(
            prepared_run / "reference_masks" / target, prepared_run / row.relative_path
        )
        if prepared_manifest["outputs"].get(row.relative_path) != row.sha256:
            raise ValueError("Reference identity differs from the preparation manifest")
        inputs.append(path)
        jobs.append(
            (
                cfg,
                source,
                target,
                row.unit_id,
                patient_of[row.unit_id],
                str(path),
                row.sha256,
                donor_data.to_dict("records"),
                ids,
                start,
                end,
            )
        )
    writer = RunWriter(
        workspace,
        cfg,
        "generate_primary",
        run_id,
        inputs=inputs,
        role=f"source_donors_and_target_references:{source}_to_{target}",
        estimand="single_draw_from_finite_search_Diffuse_generator",
        seeds={
            "streams": ["severity_donors", "perturbation_fields"],
            "key": [source, target, "patient_id", "unit_id", "draw_id"],
        },
    )
    frames, calibrations, definedness = [], [], []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        from .generation_cache import cached_generate_job

        worker = partial(cached_generate_job, workspace=workspace)
        results = pool.map(worker, jobs) if pool else map(worker, jobs)
        for index, (frame, calibration, flags) in enumerate(results):
            frames.append(frame)
            calibrations.append(calibration)
            definedness.append(flags)
            # Retain per-unit results if a later unit fails; incomplete run remains hidden.
            writer.write_table(f"units/{frame.unit_id.iloc[0]}.csv", frame)
            writer.write_table(f"calibration/{frame.unit_id.iloc[0]}.csv", calibration)
            writer.write_table(f"definedness/{frame.unit_id.iloc[0]}.csv", flags)
            print(
                f"Generated {source}->{target}: {index + 1}/{len(jobs)} units",
                flush=True,
            )
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    values = pd.concat(frames, ignore_index=True)
    all_flags = pd.concat(definedness, ignore_index=True)
    all_calibration = pd.concat(calibrations, ignore_index=True)
    if old_values is not None:
        values = pd.concat([old_values, values], ignore_index=True)
        all_flags = pd.concat([old_flags, all_flags], ignore_index=True)
        all_calibration = pd.concat(
            [old_calibration, all_calibration], ignore_index=True
        )
    writer.write_table("draw_features.csv", values)
    split = pd.read_csv(
        workspace / cfg["resolved_metadata"]["patient_split"], dtype=str
    )
    discovery = set(
        split.loc[
            split.cohort.eq(target) & split.assignment.eq("discovery"), "patient_id"
        ]
    )
    writer.write_table(
        "discovery_draw_features.csv", values.loc[values.patient_id.isin(discovery)]
    )
    writer.write_table("draw_definedness.csv", all_flags)
    writer.write_table(
        "discovery_draw_definedness.csv",
        all_flags.loc[all_flags.patient_id.isin(discovery)],
    )
    writer.write_table("calibration_attempts.csv", all_calibration)
    return writer.finish(
        {
            "source": source,
            "target": target,
            "draw_start": start,
            "draw_end": end,
            "draw_count": end,
            "workers": workers,
            "target_alternates_read": False,
            "refinement_status": "requires_construction_precision_check",
        }
    )
