from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from redesign2 import provenance
from redesign2 import simulation_pipeline as pipeline


def manifest(path, cfg, stage, extra=None):
    files = {
        p.relative_to(path).as_posix(): provenance.file_hash(p)
        for p in path.rglob("*")
        if p.is_file()
    }
    (path / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": provenance.content_hash(cfg),
                "stage": stage,
                "status": "complete",
                "outputs": files,
                "inputs": {},
                "extra": extra or {},
            }
        )
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "audit").mkdir(parents=True)
    (workspace / "config").mkdir()
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "protocol.yaml").read_text()
    )
    cfg["protocol"]["status"] = "execution_locked"
    cfg["resolved_inputs"] = {}
    cfg["resolved_metadata"] = {"patient_split": "audit/patient_split.csv"}
    cfg["simulation"]["final_mechanism_replicates_per_cell"] = 3
    cfg["simulation"]["interval_audit_datasets_per_cell"] = 3
    cfg["simulation"]["bootstrap_replicates_per_interval_audit_dataset"] = 9
    (workspace / "config/protocol.lock.yaml").write_text(yaml.safe_dump(cfg))
    (workspace / "config/protocol.lock.sha256").write_text(provenance.content_hash(cfg))
    descriptors = pd.DataFrame(
        [
            {"domain": modality, "feature_id": f"{modality}_{j}", "family": "family"}
            for modality in ["MRI", "CT"]
            for j in range(4)
        ]
    )
    descriptors.to_csv(workspace / "audit/audited_descriptor_manifest.csv", index=False)
    subjects, splits = [], []
    prepared_run, generation_run = workspace / "prepared", workspace / "generated"
    prepared_run.mkdir()
    generation_run.mkdir()
    for cohort in ["upenn", "brats_tcga", "nsclc"]:
        modality = cfg["cohorts"][cohort]["modality"]
        ids = [f"{modality}_{j}" for j in range(4)]
        frames = {"reference": [], "alternate": []}
        for patient in range(22):
            pid = f"{cohort}_{patient:03d}"
            unit = f"{cohort}_unit_{patient:03d}"
            subjects.append({"cohort_id": cohort, "patient_id": pid, "unit_id": unit})
            splits.append(
                {
                    "cohort": cohort,
                    "patient_id": pid,
                    "assignment": "discovery" if patient < 11 else "audit",
                }
            )
            for role, annotations in [
                ("reference", ["0001"]),
                ("alternate", ["0002", "0003"]),
            ]:
                for annotation in annotations:
                    for descriptor in ids:
                        frames[role].append(
                            {
                                "cohort": cohort,
                                "patient_id": pid,
                                "unit_id": unit,
                                "annotation_id": annotation,
                                "role": role,
                                "descriptor_id": descriptor,
                                "definedness": "defined",
                                "value": "EMPIRICAL_VALUES_MUST_NOT_BE_PARSED",
                            }
                        )
        (prepared_run / cohort).mkdir()
        for role, rows in frames.items():
            pd.DataFrame(rows).to_csv(
                prepared_run / cohort / f"{role}_features.csv", index=False
            )
    pd.DataFrame(subjects).to_csv(
        workspace / "audit/subjects_metadata.csv", index=False
    )
    pd.DataFrame(splits).to_csv(workspace / "audit/patient_split.csv", index=False)
    rows = []
    for subject in [row for row in subjects if row["cohort_id"] == "brats_tcga"]:
        for draw in range(100):
            rows.append(
                {
                    "cohort": "brats_tcga",
                    "patient_id": subject["patient_id"],
                    "unit_id": subject["unit_id"],
                    "draw_id": draw,
                    "batch": draw // 20,
                    **{f"MRI_{j}": "defined" for j in range(4)},
                }
            )
    pd.DataFrame(rows).to_csv(generation_run / "draw_definedness.csv", index=False)
    manifest(prepared_run, cfg, "prepare_inputs")
    manifest(
        generation_run,
        cfg,
        "generate_primary",
        {"source": "upenn", "target": "brats_tcga", "draw_count": 100},
    )
    monkeypatch.setattr(
        provenance,
        "code_state",
        lambda _: {"commit": "test_fixture", "tracked_dirty_patch_sha256": "fixture"},
    )
    monkeypatch.setattr(provenance, "environment", lambda: {"python": "fixture"})
    return workspace, cfg, prepared_run, generation_run


def test_builds_actual_reader_counts_and_audit_patient_partition_without_values(
    prepared,
):
    workspace, cfg, prepared_run, generation_run = prepared
    template, support, files, details = pipeline.build_template(
        workspace, cfg, prepared_run, "brats_tcga", generation_run=generation_run
    )
    assert template.name == "brats_tcga"
    assert len(template.patients) == 22
    assert template.alternate_counts == (2,) * 22
    assert support.reference_available.shape == (22, 4)
    assert support.real_available.shape == (44, 4)
    assert support.synthetic_available.shape == (22, 100, 4)
    assert details["empirical_value_column_read"] is False
    assert all(support.reference_available.ravel())
    audit, audit_support, _, audit_details = pipeline.build_template(
        workspace, cfg, prepared_run, "nsclc", audit_only=True
    )
    assert audit.name == "nsclc"
    assert len(set(audit.patients)) == 11
    assert all(int(patient.rsplit("_", 1)[1]) >= 11 for patient in audit.patients)
    assert audit_support.synthetic_available is None
    assert audit_details["scope"] == "audit"


def test_structural_absence_is_retained_and_missing_rows_are_not_imputed(prepared):
    workspace, cfg, prepared_run, _ = prepared
    path = prepared_run / "nsclc/alternate_features.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[0, "definedness"] = "structural_absence"
    frame.to_csv(path, index=False)
    _, support, _, details = pipeline.build_template(
        workspace, cfg, prepared_run, "nsclc"
    )
    assert support.real_available.sum() == 44 * 4 - 1
    assert details["observed_definedness_counts"]["structural_absence"] == 1
    frame.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks explicit"):
        pipeline.build_template(workspace, cfg, prepared_run, "nsclc")


def test_unknown_missingness_and_incomplete_draw_support_are_blocked(prepared):
    workspace, cfg, prepared_run, generation_run = prepared
    path = generation_run / "draw_definedness.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, "MRI_0"] = "unresolved_rule"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unknown synthetic"):
        pipeline.build_template(
            workspace, cfg, prepared_run, "brats_tcga", generation_run=generation_run
        )
    frame.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="full initial"):
        pipeline.build_template(
            workspace, cfg, prepared_run, "brats_tcga", generation_run=generation_run
        )


def fake_finite(cfg, template, support, *, regime, theta, start, stop):
    return {
        "finite_sample_mechanism": pd.DataFrame(
            [
                {
                    "template": template.name,
                    "regime": regime,
                    "theta": theta,
                    "replicate": r,
                    "h": 1.0,
                }
                for r in range(start, stop)
            ]
        ),
        "finite_mechanism_truth": pd.DataFrame(
            [
                {
                    "template": template.name,
                    "regime": regime,
                    "theta": theta,
                    "descriptor_id": d,
                    "real_population_ccc": 0.5,
                }
                for d in template.descriptor_ids
            ]
        ),
    }


def fake_coverage(cfg, template, support, *, regime, start, stop):
    names = [f"ccc:{d}" for d in template.descriptor_ids]
    names += ["utility:" + name for name in ["first", "second", "higher", "lower"]]
    names += [
        "contrast:" + name + "_minus_first" for name in ["second", "higher", "lower"]
    ]
    return {
        "coverage_repetitions": pd.DataFrame(
            [
                {
                    "template": template.name,
                    "regime": regime,
                    "replicate": r,
                    "endpoint": endpoint,
                    "population_truth": 0.5,
                    "point_estimate": 0.5,
                    "percentile_lower": 0.4,
                    "percentile_upper": 0.6,
                    "interval_issued": True,
                    "covered": True,
                    "undefined_bootstrap_fraction": 0.0,
                }
                for endpoint in names
                for r in range(start, stop)
            ]
        ),
        "coverage_fixed_sets": pd.DataFrame(
            [
                {
                    "template": template.name,
                    "regime": regime,
                    "set": "first",
                    "descriptor_id": template.descriptor_ids[0],
                    "population_mean_ccc": 0.5,
                }
            ]
        ),
    }


def test_final_finite_wrapper_contract_and_provenance_without_running_simulations(
    prepared, monkeypatch
):
    workspace, cfg, prepared_run, generation_run = prepared
    monkeypatch.setattr(pipeline, "finite_sample_tables", fake_finite)
    result = pipeline.run_finite(
        workspace,
        cfg,
        prepared_run,
        generation_run,
        "brats_tcga",
        "finite-fixture",
        workers=1,
    )
    values = pd.read_csv(result / "finite_sample_mechanism.csv")
    assert len(values) == 6 * 3
    assert set(values.template) == {"brats_tcga"}
    assert values.groupby(["regime", "theta"]).size().eq(3).all()
    record = json.loads((result / "run_manifest.json").read_text())
    assert record["stage"] == "simulation_finite"
    assert record["extra"]["required_cells_complete"] is True
    assert str(generation_run / "draw_definedness.csv") in record["inputs"]
    assert provenance.verify_run(result) == []


def test_coverage_wrapper_emits_verified_decision_and_exact_cohort_keys(
    prepared, monkeypatch
):
    workspace, cfg, prepared_run, _ = prepared
    monkeypatch.setattr(pipeline, "coverage_tables", fake_coverage)
    result = pipeline.run_coverage(
        workspace, cfg, prepared_run, "coverage-fixture", workers=1
    )
    summary = pd.read_csv(result / "coverage_summary.csv")
    assert set(summary.template) == {"nsclc", "brats_tcga"}
    assert summary.datasets.eq(3).all()
    decision = json.loads((result / "coverage_decision.json").read_text())
    assert decision["nominal_audit_intervals_allowed"] is True
    assert decision["required_cells_complete"] is True
    assert decision["shortfall_count"] == 0
    assert decision["source_table"] == "coverage_summary.csv"
    record = json.loads((result / "run_manifest.json").read_text())
    assert record["stage"] == "simulation_coverage"
    assert "coverage_decision.json" in record["outputs"]
    assert provenance.verify_run(result) == []


def test_changed_dependency_and_excess_workers_are_rejected(prepared):
    workspace, cfg, prepared_run, generation_run = prepared
    with pytest.raises(ValueError, match="1 through 12"):
        pipeline.run_finite(
            workspace,
            cfg,
            prepared_run,
            generation_run,
            "brats_tcga",
            "unused",
            workers=13,
        )
    path = prepared_run / "nsclc/reference_features.csv"
    path.write_text(path.read_text() + "corruption\n")
    with pytest.raises(ValueError, match="verification"):
        pipeline.run_coverage(workspace, cfg, prepared_run, "unused", workers=1)


def test_one_declared_coverage_shortfall_downgrades_every_audit_interval(
    prepared, monkeypatch
):
    workspace, cfg, prepared_run, _ = prepared

    def failed_coverage(*args, **kwargs):
        output = fake_coverage(*args, **kwargs)
        rows = output["coverage_repetitions"]
        selected = (
            rows.template.eq("nsclc")
            & rows.regime.eq("additive")
            & rows.endpoint.eq("utility:first")
        )
        rows.loc[selected, "covered"] = False
        rows.loc[selected, "percentile_lower"] = 0.6
        rows.loc[selected, "percentile_upper"] = 0.7
        return output

    monkeypatch.setattr(pipeline, "coverage_tables", failed_coverage)
    result = pipeline.run_coverage(
        workspace, cfg, prepared_run, "failed-coverage-fixture", workers=1
    )
    decision = json.loads((result / "coverage_decision.json").read_text())
    assert decision["required_cells_complete"] is True
    assert decision["shortfall_count"] == 1
    assert decision["nominal_audit_intervals_allowed"] is False


def test_analytic_counterexample_has_strict_order_at_both_couplings_without_grid_change(
    prepared,
):
    workspace, cfg, _, _ = prepared
    (workspace / "audit/M1_ANALYTIC_COUNTEREXAMPLE.md").write_text(
        "Analytical proof fixture; no empirical input.\n"
    )
    result = pipeline.run_counterexample(workspace, cfg, "counterexample-fixture")
    summary = pd.read_csv(result / "counterexample_summary.csv")
    values = pd.read_csv(result / "counterexample_descriptor_values.csv")
    inequalities = pd.read_csv(result / "counterexample_inequalities.csv")
    assert set(summary.theta) == {0.0, 0.9}
    assert summary.analytical_order_profile_spearman.eq(1.0).all()
    assert inequalities.log_disagreement_to_reference_ratio_margin.gt(0).all()
    assert not summary.core_grid_cell.any()
    assert not values.real_representable_ccc.equals(values.synthetic_representable_ccc)
    record = json.loads((result / "run_manifest.json").read_text())
    assert record["extra"]["seed_search_performed"] is False
    assert record["extra"]["core_simulation_grid_changed"] is False
    assert provenance.verify_run(result) == []
