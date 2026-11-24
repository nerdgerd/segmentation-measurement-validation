"""Materialize role-separated measurements before any policy construction.

This stage is permitted to inspect original contours for extraction and source
severity. Construction later receives only its declared source products and
target-reference files. No reliability, selection or utility is calculated here.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .provenance import (
    RunWriter,
    content_hash,
    file_hash,
    read_protocol,
    resolve_input_path,
)


def _prepare_job(args):
    from .adapters.prepare import prepare_unit

    repo, rows, reference, modality, ids = args
    start = time.perf_counter()
    result = prepare_unit(repo, rows, reference, modality=modality, feature_ids=ids)
    result["elapsed_seconds"] = time.perf_counter() - start
    return rows[0]["cohort_id"], rows[0]["unit_id"], result


def require_protocol_lock(workspace):
    workspace = Path(workspace).resolve()
    path = workspace / "config/protocol.lock.yaml"
    cfg = read_protocol(path)
    expected = (workspace / "config/protocol.lock.sha256").read_text().strip()
    if content_hash(cfg) != expected or cfg["protocol"]["status"] != "execution_locked":
        raise ValueError("An intact execution protocol lock is required")
    for name, digest in cfg["resolved_inputs"].items():
        source = Path(name)
        if not source.is_absolute():
            source = workspace / source
        source = resolve_input_path(source)
        if not source.is_file() or file_hash(source) != digest:
            raise ValueError(f"Locked input missing or changed: {name}")
    return cfg


def prepare_inputs(workspace, run_id, *, workers=1):
    workspace = Path(workspace).resolve()
    cfg = require_protocol_lock(workspace)
    metadata = pd.read_csv(
        workspace / "audit/contours_metadata.csv", dtype=str, keep_default_na=False
    )
    features = pd.read_csv(
        workspace / "audit/audited_descriptor_manifest.csv", keep_default_na=False
    )
    reference_path = workspace / cfg["resolved_metadata"]["ct_references"]
    split_path = workspace / cfg["resolved_metadata"]["patient_split"]
    selected = pd.read_csv(reference_path, dtype=str, keep_default_na=False)
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    jobs = []
    for (cohort, unit), group in metadata.groupby(["cohort_id", "unit_id"], sort=True):
        modality = cfg["cohorts"][cohort]["modality"]
        if modality == "MRI":
            candidates = group.loc[group.role.eq("reference"), "stable_annotation_id"]
        else:
            candidates = selected.loc[
                selected.cohort.eq(cohort) & selected.nodule_id.eq(unit),
                "reference_annotation_id",
            ]
        if len(candidates) != 1:
            raise ValueError(f"Missing unique reference identity for {cohort}/{unit}")
        ids = features.loc[features.domain.eq(modality), "feature_id"].tolist()
        jobs.append(
            (
                str(workspace.parent),
                group.to_dict("records"),
                candidates.iloc[0],
                modality,
                ids,
            )
        )
    inputs = [
        workspace / "config/protocol.lock.yaml",
        workspace / "audit/contours_metadata.csv",
        workspace / "audit/audited_descriptor_manifest.csv",
        reference_path,
        split_path,
    ]
    writer = RunWriter(
        workspace,
        cfg,
        "prepare_inputs",
        run_id,
        inputs=inputs,
        estimand="individual_contour_measurements_and_source_Dice",
        role="isolated_materialization_before_construction",
    )
    frames, severities, mask_rows, timings, contour_geometry = [], [], [], [], []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(_prepare_job, jobs) if pool else map(_prepare_job, jobs)
        for index, (cohort, unit, result) in enumerate(results):
            frames.append(result["features"])
            severities.append(result["severity"])
            mask = result["reference_mask"]
            path = writer.temp / "reference_masks" / cohort / f"{unit}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                mask=mask.mask,
                origin=np.asarray(mask.origin),
                spacing=np.asarray(mask.spacing),
            )
            mask_rows.append(
                {
                    "cohort": cohort,
                    "unit_id": unit,
                    "relative_path": path.relative_to(writer.temp).as_posix(),
                    "sha256": file_hash(path),
                    "array_sha256": result["reference_mask_array_sha256"],
                    "geometry": json.dumps(mask.metadata, sort_keys=True),
                }
            )
            contour_geometry.append(
                {
                    "cohort": cohort,
                    "unit_id": unit,
                    "all_contour_metadata": result["mask_metadata"],
                }
            )
            timings.append(
                {
                    "cohort": cohort,
                    "unit_id": unit,
                    "seconds": result["elapsed_seconds"],
                }
            )
            print(f"Prepared {index + 1}/{len(jobs)}: {cohort}/{unit}", flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    all_features = pd.concat(frames, ignore_index=True)
    all_severity = pd.concat(severities, ignore_index=True)
    for cohort in cfg["cohorts"]:
        data = all_features.loc[all_features.cohort.eq(cohort)]
        for role in ["reference", "alternate"]:
            table = data.loc[data.role.eq(role)].copy()
            # Separate files are the permissions boundary passed to subsequent processes.
            writer.write_table(f"{cohort}/{role}_features.csv", table)
            if role == "reference":
                discovery = set(
                    split.loc[
                        split.cohort.eq(cohort) & split.assignment.eq("discovery"),
                        "patient_id",
                    ]
                )
                writer.write_table(
                    f"{cohort}/discovery_reference_features.csv",
                    table.loc[table.patient_id.isin(discovery)],
                )
        from .adapters.prepare import severity_donors

        writer.write_table(
            f"{cohort}/source_severity_donors.csv",
            severity_donors(all_severity.loc[all_severity.cohort.eq(cohort)]),
        )
    writer.write_table("reference_mask_manifest.csv", pd.DataFrame(mask_rows))
    writer.write_json("isolated_audit/all_contour_geometry.json", contour_geometry)
    for cohort in cfg["cohorts"]:
        writer.write_table(
            f"{cohort}/reference_mask_manifest.csv",
            pd.DataFrame(mask_rows).loc[lambda frame: frame.cohort.eq(cohort)],
        )
    writer.write_table("extraction_timings.csv", pd.DataFrame(timings))
    return writer.finish(
        {
            "reference_units": len(jobs),
            "original_contours": len(metadata),
            "target_performance_computed": False,
            "workers": workers,
        }
    )
