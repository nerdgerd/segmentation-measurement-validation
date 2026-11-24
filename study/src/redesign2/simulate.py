"""Run locked source-independent population calculations with explicit origins."""

from pathlib import Path

import pandas as pd

from .prepare_inputs import require_protocol_lock
from .provenance import RunWriter
from .simulation_driver import population_tables


def simulate_population(workspace, run_id):
    workspace = Path(workspace).resolve()
    cfg = require_protocol_lock(workspace)
    descriptor_path = workspace / "audit/audited_descriptor_manifest.csv"
    manifest = pd.read_csv(descriptor_path, keep_default_na=False)
    tables = {}
    for cohort, specification in cfg["cohorts"].items():
        battery = manifest.loc[
            manifest.domain.eq(specification["modality"])
        ].sort_values("feature_id")
        outputs = population_tables(
            cfg,
            battery.feature_id.tolist(),
            battery.family.tolist(),
            template_key=cohort,
            template_role="audited_descriptor_identity_population_law",
        )
        for name, frame in outputs.items():
            frame = frame.copy()
            frame["cohort_template"] = cohort
            tables.setdefault(name, []).append(frame)
    ids = [f"demo_{index:02d}" for index in range(12)]
    families = [f"demo_family_{index // 4}" for index in range(12)]
    outputs = population_tables(
        cfg,
        ids,
        families,
        template_key="fixed_demonstration_v1",
        template_role="dimensionless_demonstration_not_empirical_template",
    )
    for name, frame in outputs.items():
        frame = frame.copy()
        frame["cohort_template"] = "dimensionless_demonstration"
        tables.setdefault(name, []).append(frame)
    writer = RunWriter(
        workspace,
        cfg,
        "simulate_population",
        run_id,
        inputs=[workspace / "config/protocol.lock.yaml", descriptor_path],
        role="known_population_simulation_no_empirical_performance",
        estimand="exact_Gaussian_population_disagreement_reliability_and_screening_utility",
        seeds={
            "streams": ["mechanism_final", "remedy_final"],
            "coefficient_battery": "fixed_per_template_across_case_mix_interventions",
        },
    )
    for name, pieces in tables.items():
        writer.write_table(name + ".csv", pd.concat(pieces, ignore_index=True))
    return writer.finish(
        {
            "cohort_batteries": list(cfg["cohorts"]),
            "dimensionless_demonstration": True,
            "finite_sample_mechanism_run": False,
            "coverage_audit_run": False,
            "empirical_target_performance_run": False,
        }
    )
