"""Turn audited identities and definedness flags into isolated simulation templates.

CSV readers deliberately exclude empirical descriptor values. Simulation truth
comes from the locked dimensionless law, while patient/nodule/reader structure
and missingness come from separately verified preparation and generation files.
Each worker receives only that structure, Boolean masks, and the locked protocol.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_inputs import require_protocol_lock
from .provenance import RunWriter, content_hash, verify_run
from .simulation import Template
from .simulation_repetitions import (
    SimulationSupport,
    finite_sample_tables,
    coverage_tables,
    summarize_coverage,
)


DEFINEDNESS = {
    "defined",
    "structural_absence",
    "below_intercompartment_voxel_support",
    "insufficient_angular_support",
    "below_50_voxel_shape_support",
    "numerical_undefined",
}
FLAG_COLUMNS = [
    "cohort",
    "patient_id",
    "unit_id",
    "annotation_id",
    "role",
    "descriptor_id",
    "definedness",
]


def _worker_count(workers):
    if (
        not isinstance(workers, int)
        or isinstance(workers, bool)
        or not 1 <= workers <= 12
    ):
        raise ValueError("Simulation workers must be an integer from 1 through 12")
    return workers


def _locked_cfg(workspace, cfg):
    locked = require_protocol_lock(workspace)
    if content_hash(locked) != content_hash(cfg):
        raise ValueError("Simulation configuration does not match the execution lock")
    return locked


def _verified_stage(path, cfg, expected_stage):
    path = Path(path).resolve()
    failures = verify_run(path)
    if failures:
        raise ValueError(
            f"Simulation dependency failed verification: {path}: {failures}"
        )
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["protocol_sha256"] != content_hash(cfg)
        or manifest["stage"] != expected_stage
    ):
        raise ValueError(
            f"Simulation dependency has the wrong protocol or stage: {path}"
        )
    if manifest.get("status", "complete") != "complete":
        raise ValueError(f"Simulation dependency is incomplete: {path}")
    return manifest


def _read_flags(path, cohort, role):
    frame = pd.read_csv(path, usecols=FLAG_COLUMNS, dtype=str, keep_default_na=False)
    if not frame.cohort.eq(cohort).all() or not frame.role.eq(role).all():
        raise ValueError(f"Prepared flags contain another cohort or role: {path}")
    if not set(frame.definedness).issubset(DEFINEDNESS):
        raise ValueError(f"Unknown descriptor missingness rule in {path}")
    keys = ["unit_id", "annotation_id", "descriptor_id"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicated prepared contour/descriptor identities: {path}")
    return frame


def build_template(
    workspace, cfg, prepared_run, cohort, *, generation_run=None, audit_only=False
):
    """Read only metadata and definedness; missing rows are errors, not imputed flags."""
    workspace, prepared_run = Path(workspace).resolve(), Path(prepared_run).resolve()
    if cohort not in cfg["cohorts"]:
        raise ValueError("Unknown locked cohort")
    manifest_path = workspace / "audit" / "audited_descriptor_manifest.csv"
    subjects_path = workspace / "audit" / "subjects_metadata.csv"
    split_path = workspace / cfg["resolved_metadata"]["patient_split"]
    reference_path = prepared_run / cohort / "reference_features.csv"
    alternate_path = prepared_run / cohort / "alternate_features.csv"
    descriptors = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    descriptors = descriptors.loc[
        descriptors.domain.eq(cfg["cohorts"][cohort]["modality"])
    ]
    if (
        descriptors.empty
        or descriptors.feature_id.duplicated().any()
        or descriptors.family.eq("").any()
    ):
        raise ValueError("Descriptor identities/families are missing or ambiguous")
    ids, families = tuple(descriptors.feature_id), tuple(descriptors.family)
    subjects = pd.read_csv(subjects_path, dtype=str, keep_default_na=False)
    subjects = subjects.loc[subjects.cohort_id.eq(cohort)].sort_values(
        ["patient_id", "unit_id"]
    )
    if subjects.empty or subjects.unit_id.duplicated().any():
        raise ValueError("Patient/unit metadata are absent or ambiguous")
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    split = split.loc[split.cohort.eq(cohort)]
    if split.patient_id.duplicated().any() or set(split.patient_id) != set(
        subjects.patient_id
    ):
        raise ValueError("Locked split does not match the original patient identities")
    if not set(split.assignment).issubset({"discovery", "audit"}):
        raise ValueError("Unknown patient split assignment")
    all_units = set(subjects.unit_id)
    reference = _read_flags(reference_path, cohort, "reference")
    alternate = _read_flags(alternate_path, cohort, "alternate")
    for frame in [reference, alternate]:
        if set(frame.unit_id) != all_units or set(frame.descriptor_id) != set(ids):
            raise ValueError(
                "Prepared flags do not cover the audited units and descriptor battery"
            )
        patients = dict(subjects[["unit_id", "patient_id"]].to_numpy())
        if not frame.patient_id.eq(frame.unit_id.map(patients)).all():
            raise ValueError("Prepared flags disagree with patient/unit correspondence")
        if not frame.groupby(["unit_id", "annotation_id"]).size().eq(len(ids)).all():
            raise ValueError(
                "A prepared contour lacks explicit descriptor definedness rows"
            )
    if not reference.groupby("unit_id").annotation_id.nunique().eq(1).all():
        raise ValueError("Each unit requires exactly one reference contour")
    reference_annotations = set(zip(reference.unit_id, reference.annotation_id))
    if reference_annotations.intersection(
        zip(alternate.unit_id, alternate.annotation_id)
    ):
        raise ValueError("A reference contour is duplicated as an observed alternate")
    if audit_only:
        audit_patients = set(split.loc[split.assignment.eq("audit"), "patient_id"])
        subjects = subjects.loc[subjects.patient_id.isin(audit_patients)]
    units = tuple(subjects.unit_id)
    reference = reference.loc[reference.unit_id.isin(units)]
    alternate = alternate.loc[alternate.unit_id.isin(units)]
    alternate_ids = {
        unit: sorted(
            alternate.loc[alternate.unit_id.eq(unit), "annotation_id"].unique()
        )
        for unit in units
    }
    if any(not annotations for annotations in alternate_ids.values()):
        raise ValueError("A selected reference unit has no observed alternate")
    template = Template(
        cohort,
        ids,
        families,
        tuple(subjects.patient_id),
        units,
        tuple(len(alternate_ids[unit]) for unit in units),
    )
    template.validate()
    reference_index = pd.MultiIndex.from_product(
        [units, ids], names=["unit_id", "descriptor_id"]
    )
    reference_flags = reference.set_index(
        ["unit_id", "descriptor_id"]
    ).definedness.reindex(reference_index)
    real_index = pd.MultiIndex.from_tuples(
        [
            (unit, annotation, descriptor)
            for unit in units
            for annotation in alternate_ids[unit]
            for descriptor in ids
        ],
        names=["unit_id", "annotation_id", "descriptor_id"],
    )
    real_flags = alternate.set_index(
        ["unit_id", "annotation_id", "descriptor_id"]
    ).definedness.reindex(real_index)
    if reference_flags.isna().any() or real_flags.isna().any():
        raise ValueError("Missing explicit metadata-defined observation support")
    reference_available = (
        reference_flags.eq("defined").to_numpy().reshape(len(units), len(ids))
    )
    real_available = (
        real_flags.eq("defined")
        .to_numpy()
        .reshape(sum(template.alternate_counts), len(ids))
    )
    inputs = [
        manifest_path,
        subjects_path,
        split_path,
        reference_path,
        alternate_path,
        prepared_run / "run_manifest.json",
    ]
    details = {
        "cohort": cohort,
        "scope": "audit" if audit_only else "full",
        "patients": len(set(template.patients)),
        "reference_units": len(units),
        "observed_alternates": sum(template.alternate_counts),
        "descriptor_count": len(ids),
        "source_of_truth": "locked_dimensionless_simulation_law",
        "empirical_value_column_read": False,
        "reference_definedness_counts": reference_flags.value_counts().to_dict(),
        "observed_definedness_counts": real_flags.value_counts().to_dict(),
    }
    synthetic_available = None
    if generation_run is not None:
        if audit_only:
            raise ValueError("Coverage templates do not consume generated target draws")
        generation_run = Path(generation_run).resolve()
        draw_path = generation_run / "draw_definedness.csv"
        draws = pd.read_csv(
            draw_path,
            usecols=["cohort", "patient_id", "unit_id", "draw_id", "batch", *ids],
            dtype={"cohort": str, "patient_id": str, "unit_id": str},
            keep_default_na=False,
        )
        if not draws.cohort.eq(cohort).all() or set(draws.unit_id) != set(units):
            raise ValueError("Generated draw flags do not match the target template")
        patient_of = dict(zip(units, template.patients, strict=True))
        if not draws.patient_id.eq(draws.unit_id.map(patient_of)).all():
            raise ValueError(
                "Generated flags have inconsistent patient/unit correspondence"
            )
        if (
            draws.duplicated(["unit_id", "draw_id"]).any()
            or not np.equal(draws.draw_id, np.floor(draws.draw_id)).all()
            or not np.equal(draws.batch, draws.draw_id // 20).all()
        ):
            raise ValueError(
                "Generated draw identities/batches are duplicated or invalid"
            )
        if not set(draws[list(ids)].to_numpy().ravel()).issubset(DEFINEDNESS):
            raise ValueError("Unknown synthetic descriptor missingness rule")
        count = cfg["synthetic"]["draw_count_initial"]
        selected = draws.loc[draws.draw_id.ge(0) & draws.draw_id.lt(count)]
        index = pd.MultiIndex.from_product(
            [units, range(count)], names=["unit_id", "draw_id"]
        )
        flags = selected.set_index(["unit_id", "draw_id"])[list(ids)].reindex(index)
        if flags.isna().any().any():
            raise ValueError(
                "Finalized generation lacks the full initial synthetic draw support"
            )
        synthetic_available = (
            flags.eq("defined").to_numpy().reshape(len(units), count, len(ids))
        )
        details.update(
            {
                "synthetic_definedness_draws_per_unit": count,
                "synthetic_definedness_scope": "initial_draw_indices_of_finalized_generated_pool",
                "synthetic_definedness_counts": pd.Series(flags.to_numpy().ravel())
                .value_counts()
                .to_dict(),
            }
        )
        inputs.extend([draw_path, generation_run / "run_manifest.json"])
    support = SimulationSupport(
        reference_available, real_available, synthetic_available
    )
    support.validate(
        template,
        require_synthetic=generation_run is not None,
        draw_count=cfg["synthetic"]["draw_count_initial"],
    )
    details.update(
        template_sha256=template.identity, availability_sha256=support.identity
    )
    return template, support, inputs, details


def _ranges(total, size=100):
    return [(start, min(start + size, total)) for start in range(0, total, size)]


def _finite_job(job):
    cfg, template, support, regime, theta, start, stop = job
    return (regime, theta, start, stop), finite_sample_tables(
        cfg, template, support, regime=regime, theta=theta, start=start, stop=stop
    )


def _coverage_job(job):
    cfg, template, support, regime, start, stop = job
    return (template.name, regime, start, stop), coverage_tables(
        cfg, template, support, regime=regime, start=start, stop=stop
    )


def _execute(jobs, worker, workers):
    if workers == 1:
        yield from map(worker, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def _template_records(template, details):
    return {
        **details,
        "descriptor_ids": template.descriptor_ids,
        "families": template.families,
        "patient_ids_by_unit": template.patients,
        "unit_ids": template.nodules,
        "alternate_counts_by_unit": template.alternate_counts,
    }


def run_finite(
    workspace, cfg, prepared_run, generation_run, cohort, run_id, *, workers=12
):
    workers = _worker_count(workers)
    workspace = Path(workspace).resolve()
    cfg = _locked_cfg(workspace, cfg)
    _verified_stage(prepared_run, cfg, "prepare_inputs")
    generation = _verified_stage(generation_run, cfg, "generate_primary")
    if generation["extra"]["target"] != cohort or not any(
        d["target"] == cohort and d["source"] == generation["extra"]["source"]
        for d in cfg["directions"]
    ):
        raise ValueError("Generation has the wrong locked source-target direction")
    template, support, inputs, details = build_template(
        workspace, cfg, prepared_run, cohort, generation_run=generation_run
    )
    count = cfg["simulation"]["final_mechanism_replicates_per_cell"]
    cells = list(
        product(
            cfg["simulation"]["core_regimes"], cfg["simulation"]["finite_theta_grid"]
        )
    )
    jobs = [
        (cfg, template, support, regime, theta, start, stop)
        for regime, theta in cells
        for start, stop in _ranges(count)
    ]
    writer = RunWriter(
        workspace,
        cfg,
        "simulation_finite",
        run_id,
        inputs=[workspace / "config/protocol.lock.yaml", *inputs],
        estimand="finite_patient_sampling_of_fixed_dimensionless_descriptor_battery",
        role="isolated_metadata_and_definedness_simulation",
        seeds={
            "stream": "mechanism_final",
            "coefficient_template": cohort,
            "patient_key": ["patient_fields", template.identity, "absolute_replicate"],
        },
    )
    writer.write_json("template.json", _template_records(template, details))
    repetitions, truths = [], []
    for finished, (key, tables) in enumerate(
        _execute(jobs, _finite_job, workers), start=1
    ):
        regime, theta, start, stop = key
        writer.write_table(
            f"batches/{regime}_theta{theta}_{start:04d}_{stop:04d}.csv",
            tables["finite_sample_mechanism"],
        )
        repetitions.append(tables["finite_sample_mechanism"])
        truths.append(tables["finite_mechanism_truth"])
        print(f"Finite simulation {cohort}: {finished}/{len(jobs)} batches", flush=True)
    result = (
        pd.concat(repetitions, ignore_index=True)
        .sort_values(["regime", "theta", "replicate"])
        .reset_index(drop=True)
    )
    if result.duplicated(["regime", "theta", "replicate"]).any():
        raise ValueError("Overlapping finite simulation replicate batches")
    for regime, theta in cells:
        observed = result.loc[
            result.regime.eq(regime) & result.theta.eq(theta), "replicate"
        ]
        if observed.tolist() != list(range(count)):
            raise ValueError("Finite simulation cell is incomplete")
    truth = (
        pd.concat(truths, ignore_index=True)
        .drop_duplicates()
        .sort_values(["regime", "theta", "descriptor_id"])
    )
    if truth.duplicated(["template", "regime", "theta", "descriptor_id"]).any():
        raise ValueError("Population truth changed across finite batches")
    writer.write_table("finite_sample_mechanism.csv", result)
    writer.write_table("finite_mechanism_truth.csv", truth)
    return writer.finish(
        {
            "cohort": cohort,
            "cells": len(cells),
            "replicates_per_cell": count,
            "required_cells_complete": True,
            "workers": workers,
            "empirical_performance_computed": False,
            "template": details,
        }
    )


def run_coverage(workspace, cfg, prepared_run, run_id, *, workers=12):
    workers = _worker_count(workers)
    workspace = Path(workspace).resolve()
    cfg = _locked_cfg(workspace, cfg)
    _verified_stage(prepared_run, cfg, "prepare_inputs")
    cohorts = cfg["simulation"]["interval_audit_templates"]
    regimes = cfg["simulation"]["interval_audit_regimes"]
    count = cfg["simulation"]["interval_audit_datasets_per_cell"]
    templates, inputs, jobs = {}, [workspace / "config/protocol.lock.yaml"], []
    for cohort in cohorts:
        template, support, files, details = build_template(
            workspace, cfg, prepared_run, cohort, audit_only=True
        )
        templates[cohort] = _template_records(template, details)
        inputs.extend(files)
        for regime in regimes:
            jobs.extend(
                (cfg, template, support, regime, start, stop)
                for start, stop in _ranges(count)
            )
    writer = RunWriter(
        workspace,
        cfg,
        "simulation_coverage",
        run_id,
        inputs=list(dict.fromkeys(inputs)),
        estimand="coverage_of_observed_CCC_and_fixed_selection_utility_on_independent_audit_patients",
        role="isolated_metadata_and_definedness_simulation",
        seeds={
            "stream": "interval_audit",
            "coefficient_templates": list(cohorts),
            "bootstrap_key": [
                "coverage_bootstrap",
                "template_sha256",
                "regime",
                "absolute_replicate",
            ],
        },
    )
    writer.write_json("templates.json", templates)
    repetitions, selected_sets = [], []
    for finished, (key, tables) in enumerate(
        _execute(jobs, _coverage_job, workers), start=1
    ):
        cohort, regime, start, stop = key
        writer.write_table(
            f"batches/{cohort}_{regime}_{start:04d}_{stop:04d}.csv",
            tables["coverage_repetitions"],
        )
        repetitions.append(tables["coverage_repetitions"])
        selected_sets.append(tables["coverage_fixed_sets"])
        print(f"Coverage simulation: {finished}/{len(jobs)} batches", flush=True)
    result = (
        pd.concat(repetitions, ignore_index=True)
        .sort_values(["template", "regime", "endpoint", "replicate"])
        .reset_index(drop=True)
    )
    summary = summarize_coverage(
        result,
        nominal=cfg["uncertainty"]["nominal_pointwise_level"],
        material_shortfall=cfg["uncertainty"]["material_coverage_shortfall"],
    )
    expected_cells = set(product(cohorts, regimes))
    present_cells = set(zip(summary.template, summary.regime))
    complete = expected_cells == present_cells
    for (cohort, regime, endpoint), group in result.groupby(
        ["template", "regime", "endpoint"]
    ):
        if group.replicate.tolist() != list(range(count)):
            complete = False
    for cohort, regime in expected_cells:
        expected_endpoints = {"ccc:" + d for d in templates[cohort]["descriptor_ids"]}
        expected_endpoints |= {
            "utility:" + name for name in ["first", "second", "higher", "lower"]
        }
        expected_endpoints |= {
            "contrast:" + name + "_minus_first"
            for name in ["second", "higher", "lower"]
        }
        rows = summary.loc[summary.template.eq(cohort) & summary.regime.eq(regime)]
        if (
            set(rows.endpoint) != expected_endpoints
            or not rows.datasets.eq(count).all()
        ):
            complete = False
    fixed_sets = (
        pd.concat(selected_sets, ignore_index=True)
        .drop_duplicates()
        .sort_values(["template", "regime", "set", "descriptor_id"])
    )
    if fixed_sets.duplicated(["template", "regime", "set", "descriptor_id"]).any():
        raise ValueError("Coverage fixed-set truths changed across batches")
    shortfall_count = int(summary.material_coverage_shortfall.sum())
    decision = {
        "nominal_audit_intervals_allowed": bool(complete and shortfall_count == 0),
        "required_cells_complete": bool(complete),
        "shortfall_count": shortfall_count,
        "status": "nominal_pointwise_intervals_allowed"
        if complete and shortfall_count == 0
        else "downgrade_all_audit_intervals_to_sampling_stability",
        "required_cells": [
            {"template": cohort, "regime": regime}
            for cohort, regime in sorted(expected_cells)
        ],
        "datasets_per_cell": count,
        "decision_rule": "any_required_endpoint_below_nominal_minus_0.02_or_incomplete_audit_downgrades_all_empirical_audit_intervals",
        "source_table": "coverage_summary.csv",
        "source_key_columns": ["template", "regime", "endpoint"],
        "limitations": "Conservative reporting rule under four declared mechanisms; passing is not universal calibration.",
    }
    writer.write_table("coverage_repetitions.csv", result)
    writer.write_table("coverage_summary.csv", summary)
    writer.write_table("coverage_fixed_sets.csv", fixed_sets)
    writer.write_json("coverage_decision.json", decision)
    return writer.finish(
        {
            "required_cells_complete": bool(complete),
            "cells": len(expected_cells),
            "datasets_per_cell": count,
            "workers": workers,
            "empirical_performance_computed": False,
            "coverage_decision": decision,
        }
    )


def run_counterexample(workspace, cfg, run_id):
    """Realize the M1 existence proof using its fixed fields, without altering a grid."""
    from scipy.special import expit
    from scipy.stats import rankdata, spearmanr
    from .simulation import coefficient_battery, demonstration_template, make_laws

    workspace = Path(workspace).resolve()
    cfg = _locked_cfg(workspace, cfg)
    template = demonstration_template()
    battery = coefficient_battery(
        cfg,
        template.descriptor_ids,
        template.families,
        template_key=template.name,
        stream="mechanism_final",
    )
    g = battery.reference.to_numpy()
    order = np.argsort(g)
    differences = np.diff(g[order])
    if not np.all(differences > 0):
        raise ValueError(
            "The fixed demonstration reference coefficients are tied; this construction is undefined"
        )
    real, syn0 = make_laws(battery, h=0, theta=0.0, regime="additive")
    _, syn9 = make_laws(battery, h=0, theta=0.9, regime="additive")
    logs = {
        "real": 2 * np.log(real.residual_sd),
        "synthetic_theta0": 2 * np.log(syn0.residual_sd),
        "synthetic_theta0p9": 2 * np.log(syn9.residual_sd),
    }
    thresholds = {
        name: np.diff(values[order]) / (2 * differences)
        for name, values in logs.items()
    }
    H = max(0.0, max(float(values.max()) for values in thresholds.values()))
    h = H + 1.0
    rows, inequalities, summaries = [], [], []
    for name, values in logs.items():
        log_ratio = values - 2 * h * g
        margins = -np.diff(log_ratio[order])
        if not np.all(margins > 0):
            raise ValueError(
                "The derived strict inequalities were lost at floating-point precision"
            )
        for j in range(len(order) - 1):
            a, b = order[j : j + 2]
            inequalities.append(
                {
                    "process": name,
                    "lower_g_descriptor": template.descriptor_ids[a],
                    "higher_g_descriptor": template.descriptor_ids[b],
                    "reference_g_gap": differences[j],
                    "adjacent_threshold": thresholds[name][j],
                    "global_H": H,
                    "constructed_h": h,
                    "log_disagreement_to_reference_ratio_margin": margins[j],
                    "strict_order_verified": True,
                }
            )
    real_log_ratio = logs["real"] - 2 * h * g
    real_ccc = expit(np.log(2.0) - real_log_ratio)
    real_rank = rankdata(-real_log_ratio)
    for theta, process in [(0.0, "synthetic_theta0"), (0.9, "synthetic_theta0p9")]:
        synthetic_log_ratio = logs[process] - 2 * h * g
        synthetic_rank = rankdata(-synthetic_log_ratio)
        synthetic_ccc = expit(np.log(2.0) - synthetic_log_ratio)
        exact_order_association = float(
            spearmanr(-real_log_ratio, -synthetic_log_ratio).statistic
        )
        rounded_association = float(spearmanr(real_ccc, synthetic_ccc).statistic)
        for j, descriptor in enumerate(template.descriptor_ids):
            rows.append(
                {
                    "descriptor_id": descriptor,
                    "family": template.families[j],
                    "theta": theta,
                    "reference_log_dispersion_coefficient": g[j],
                    "constructed_h": h,
                    "reference_log_variance": 2 * h * g[j],
                    "real_log_disagreement_variance": logs["real"][j],
                    "synthetic_log_disagreement_variance": logs[process][j],
                    "real_log_disagreement_to_reference_variance": real_log_ratio[j],
                    "synthetic_log_disagreement_to_reference_variance": synthetic_log_ratio[
                        j
                    ],
                    "real_analytical_ccc_order": real_rank[j],
                    "synthetic_analytical_ccc_order": synthetic_rank[j],
                    "real_representable_ccc": real_ccc[j],
                    "synthetic_representable_ccc": synthetic_ccc[j],
                }
            )
        summaries.append(
            {
                "template": template.name,
                "theta": theta,
                "descriptor_count": len(g),
                "global_H": H,
                "constructed_h": h,
                "analytical_order_profile_spearman": exact_order_association,
                "representable_ccc_profile_spearman": rounded_association,
                "real_representable_ccc_distinct_values": len(np.unique(real_ccc)),
                "synthetic_representable_ccc_distinct_values": len(
                    np.unique(synthetic_ccc)
                ),
                "core_grid_cell": False,
                "same_descriptor_ccc_values_claimed": False,
            }
        )
    proof = workspace / "audit/M1_ANALYTIC_COUNTEREXAMPLE.md"
    writer = RunWriter(
        workspace,
        cfg,
        "population_counterexample",
        run_id,
        inputs=[workspace / "config/protocol.lock.yaml", proof],
        estimand="analytical_equal_profile_association_under_distinct_residual_log_scale_couplings",
        role="source_independent_fixed_coefficient_existence_proof",
        seeds={
            "stream": "mechanism_final",
            "template_key": template.name,
            "coefficient_key": [
                "coefficient",
                template.name,
                "kind",
                "family_or_descriptor",
                "stable_id",
            ],
        },
    )
    writer.write_table("counterexample_descriptor_values.csv", pd.DataFrame(rows))
    writer.write_table("counterexample_inequalities.csv", pd.DataFrame(inequalities))
    writer.write_table("counterexample_summary.csv", pd.DataFrame(summaries))
    writer.write_table("counterexample_coefficient_fields.csv", battery)
    writer.write_text("analytical_proof.md", proof.read_text(encoding="utf-8"))
    return writer.finish(
        {
            "core_simulation_grid_changed": False,
            "seed_search_performed": False,
            "empirical_values_used": False,
            "construction_rule": "h=max(0,all_adjacent_thresholds)+1",
            "interpretation": "Identical CCC-profile association at two different imposed coupling parameters, not identical descriptor CCC values.",
        }
    )
