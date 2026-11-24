"""Evaluation cannot improve a screen by changing its support or using audit adaptation."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from redesign2.construction import MatrixMeasurements
from redesign2.empirical_math import aggregate_patient_moments
from redesign2.evaluate import (
    evaluate_direction,
    evaluate_from_frames,
    matched_diagnostic_matrices,
    source_residual_diagnostics,
    _coverage_nominal_permission,
)
from redesign2.moments import JointMoments
from redesign2.policies import construct_screens
from redesign2.provenance import content_hash, file_hash, verify_run
from redesign2.transport import SourceModel


def config(replicates=12):
    return {
        "features": {
            "min_evaluation_independent_patients": 10,
            "min_evaluation_weighted_support": 0.9,
            "min_construction_weighted_availability": 0.9,
            "min_source_independent_patients": 10,
        },
        "remedy": {"residual_variance_negative_roundoff_relative_tolerance": 1e-10},
        "uncertainty": {
            "empirical_bootstrap_replicates": replicates,
            "nominal_pointwise_level": 0.95,
            "max_undefined_bootstrap_fraction_for_nominal_interval": 0.05,
        },
        "randomness": {
            "master_seed": 17,
            "namespace": "artificial_evaluation_test",
            "streams": {"empirical_bootstrap": "bootstrap"},
        },
        "directions": [{"source": "source", "target": "target"}],
    }


def frames(missing_selected=False):
    descriptors = tuple(f"d{j}" for j in range(4))
    refs, alts, draws, flags = [], [], [], []
    for patient in range(12):
        pid = f"p{patient:02d}"
        for index, descriptor in enumerate(descriptors):
            value = (patient + 1) * (index + 1.0)
            refs.append(
                {
                    "cohort": "target",
                    "patient_id": pid,
                    "unit_id": pid,
                    "annotation_id": "ref",
                    "role": "reference",
                    "descriptor_id": descriptor,
                    "value": value,
                    "definedness": "defined",
                }
            )
            absent = missing_selected and index == 0 and patient < 2
            alts.append(
                {
                    "cohort": "target",
                    "patient_id": pid,
                    "unit_id": pid,
                    "annotation_id": "alt",
                    "role": "alternate",
                    "descriptor_id": descriptor,
                    "value": nan if absent else value + 0.1 * (patient % 3 - 1),
                    "definedness": "structural_absence" if absent else "defined",
                }
            )
        for draw in range(4):
            key = {
                "cohort": "target",
                "patient_id": pid,
                "unit_id": pid,
                "draw_id": draw,
                "batch": 0,
            }
            draws.append(
                {
                    **key,
                    **{
                        descriptor: (patient + 1) * (index + 1.0) + 0.25 * (draw - 1.5)
                        for index, descriptor in enumerate(descriptors)
                    },
                }
            )
            flags.append(
                {**key, **{descriptor: "defined" for descriptor in descriptors}}
            )
    maps = {
        "source_ccc": {"d0": 0.99, "d1": 0.9, "d2": 0.8, "d3": 0.7},
        "synthetic_ccc": {"d0": 0.7, "d1": 0.99, "d2": 0.9, "d3": 0.8},
        "error_transport": {"d0": 0.7, "d1": 0.8, "d2": 0.99, "d3": 0.9},
    }
    models = {
        descriptor: SourceModel(0, 1, 0.1, 0, 1, 12) for descriptor in descriptors
    }
    return dict(
        source="source",
        target="target",
        universe=descriptors,
        selections=construct_screens(**maps),
        source_models=models,
        score_maps=maps,
        target_reference=pd.DataFrame(refs),
        target_alternate=pd.DataFrame(alts),
        draw_features=pd.DataFrame(draws),
        draw_definedness=pd.DataFrame(flags),
        descriptor_families={"d0": "a", "d1": "a", "d2": "b", "d3": "b"},
    )


nan = float("nan")


def test_all_commitments_gate_precedes_target_file_access(monkeypatch, tmp_path):
    def blocked(*_args, **_kwargs):
        raise PermissionError("all eight required")

    monkeypatch.setattr("redesign2.evaluate.require_all_constructions", blocked)
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *_a, **_k: pytest.fail(
            "No target table may open before commitment gate"
        ),
    )
    with pytest.raises(PermissionError, match="eight"):
        evaluate_direction(
            tmp_path,
            config(),
            tmp_path,
            tmp_path,
            tmp_path,
            [],
            "source",
            "target",
            "full",
            "run",
        )


def test_unevaluable_selected_descriptor_invalidates_exact_selection():
    result = evaluate_from_frames(
        config(), scope="audit", **frames(missing_selected=True)
    )
    utility = result["secondary_selection_metrics"].set_index("policy")
    assert np.isnan(utility.loc["source_ccc", "utility"])
    assert (
        utility.loc["source_ccc", "utility_reason"] == "unevaluable_selected_descriptor"
    )
    assert np.isfinite(utility.loc["synthetic_ccc", "utility"])
    bootstrap = result["bootstrap_replicates"]
    assert bootstrap["utility:source_ccc"].isna().all()
    assert result["observed_moments"].set_index("descriptor_id").loc[
        "d0", "support_original_weighted_support"
    ] == pytest.approx(10 / 12)


def test_audit_bootstrap_keeps_discovery_selections_and_downgrades_uncovered_diagnostics():
    result = evaluate_from_frames(
        config(), scope="audit", coverage_nominal_allowed=True, **frames()
    )
    frequency = result["bootstrap_selection_frequency"]
    for policy, chosen in (
        ("source_ccc", "d0"),
        ("synthetic_ccc", "d1"),
        ("error_transport", "d2"),
    ):
        selected = frequency.loc[frequency.policy.eq(policy)].set_index("descriptor_id")
        assert selected.loc[chosen, "selected_replicates"] == 12
        assert selected.selected_replicates.sum() == 12
    summary = result["bootstrap_summary"].set_index("statistic_id")
    assert (
        summary.loc["utility:synthetic_ccc", "interval_kind"]
        == "pointwise_nominal_percentile_interval"
    )
    assert (
        summary.loc["profile_spearman:synthetic_ccc", "interval_kind"]
        == "bootstrap_sampling_stability_range"
    )


def test_full_bootstrap_recomputes_target_adaptive_scores_from_reference_pool():
    result = evaluate_from_frames(config(), scope="full", **frames())
    selected = result["bootstrap_selection_frequency"]
    # Equal source conditional disagreement makes the largest reference variance
    # rank first in each resample; the discovery fixture originally chose d2.
    transport = selected.loc[selected.policy.eq("error_transport")].set_index(
        "descriptor_id"
    )
    assert transport.loc["d3", "selected_replicates"] == 12
    assert transport.loc["d2", "selected_replicates"] == 0
    assert set(result["bootstrap_summary"].interval_kind) == {
        "bootstrap_sampling_stability_range"
    }


def test_full_reselection_does_not_restore_an_undefined_original_utility_interval():
    fixture = frames()
    alternate = fixture["target_alternate"]
    missing = alternate.descriptor_id.eq("d2") & alternate.patient_id.isin(
        ["p00", "p01"]
    )
    alternate.loc[missing, "value"] = np.nan
    alternate.loc[missing, "definedness"] = "structural_absence"
    result = evaluate_from_frames(config(), scope="full", **fixture)
    point = (
        result["secondary_selection_metrics"].set_index("policy").loc["error_transport"]
    )
    assert point.utility_reason == "unevaluable_selected_descriptor"
    assert result["bootstrap_replicates"]["utility:error_transport"].notna().all()
    summary = (
        result["bootstrap_summary"]
        .set_index("statistic_id")
        .loc["utility:error_transport"]
    )
    assert summary.interval_kind == "not_issued"
    assert summary.reason == "original_statistic_undefined"
    assert np.isnan(summary.interval_lower)
    assert np.isfinite(summary.available_distribution_lower)


def test_lofo_recomputes_remaining_budget_without_losing_denominator():
    result = evaluate_from_frames(config(), scope="audit", **frames())
    sensitivity = result["leave_one_family_out"]
    assert set(sensitivity.omitted_family) == {"a", "b"}
    assert set(sensitivity.universe_count) == {2}
    assert set(sensitivity.full_construction_universe_count) == {4}
    assert set(sensitivity.budget) == {1}


def test_matched_diagnostics_condition_on_units_without_reader_duplication():
    ids = ("descriptor",)
    real_keys = pd.DataFrame(
        {
            "patient_id": ["p", "p", "p", "q"],
            "unit_id": ["u", "u", "v", "w"],
            "observation_id": ["r1", "r2", "r1", "r1"],
        }
    )
    syn_keys = pd.DataFrame(
        {
            "patient_id": ["p", "p", "p", "p", "q", "q"],
            "unit_id": ["u", "u", "v", "v", "w", "w"],
            "observation_id": ["s1", "s2"] * 3,
        }
    )
    real = MatrixMeasurements(
        ids,
        real_keys,
        np.array([[1.0], [1.0], [50.0], [3.0]]),
        np.array([[2.0], [2.0], [51.0], [4.0]]),
        np.array([[True], [False], [False], [True]]),
    )
    synthetic = MatrixMeasurements(
        ids,
        syn_keys,
        np.array([[1.0], [1.0], [50.0], [50.0], [3.0], [3.0]]),
        np.array([[2.0], [2.0], [51.0], [51.0], [4.0], [4.0]]),
        np.ones((6, 1), bool),
    )
    r, s = matched_diagnostic_matrices(real, synthetic)
    rm = aggregate_patient_moments(
        r.patient_moments(), min_independent_patients=1, min_weighted_support=0
    )
    sm = aggregate_patient_moments(
        s.patient_moments(), min_independent_patients=1, min_weighted_support=0
    )
    assert rm.moments[0].reference_mean == sm.moments[0].reference_mean == 2
    assert s.reference.shape[0] == 6
    assert s.eligible[:, 0].tolist() == [True, True, False, False, True, True]


def test_disagreement_diagnostics_retain_locked_discovery_predictions_and_raw_feature_moments():
    fixture = frames()
    locked = {
        descriptor: JointMoments(50, 20, 3, 4, 2, 13, 0.7)
        for descriptor in fixture["universe"]
    }
    result = evaluate_from_frames(
        config(2), scope="audit", locked_predictions=locked, **fixture
    )
    transported = result["transport_disagreement"].set_index("descriptor_id")
    assert set(transported.predicted_disagreement_mse) == {13}
    assert set(transported.predicted_bias) == {3}
    assert set(transported.predicted_reference_slope) == {0.1}
    assert set(transported.prediction_origin) == {"locked_discovery_prediction"}
    observed = result["observed_moments"].set_index("descriptor_id")
    assert np.allclose(transported.observed_disagreement_mse, observed.disagreement_mse)
    synthetic = result["descriptor_disagreement"].set_index("descriptor_id")
    matched = result["matched_synthetic_moments"].set_index("descriptor_id")
    assert np.allclose(synthetic.predicted_disagreement_mse, matched.disagreement_mse)
    assert np.allclose(synthetic.synthetic_construction_score, [0.7, 0.99, 0.9, 0.8])


def test_source_residuals_use_fixed_fit_and_conditional_hierarchical_weights():
    references = pd.DataFrame(
        [
            {
                "cohort": "source",
                "patient_id": p,
                "unit_id": u,
                "annotation_id": "ref",
                "role": "reference",
                "descriptor_id": "d",
                "value": x,
                "definedness": "defined",
            }
            for p, u, x in [("p", "u", 1), ("p", "v", 3), ("q", "w", 5)]
        ]
    )
    alternates = pd.DataFrame(
        [
            {
                "cohort": "source",
                "patient_id": p,
                "unit_id": u,
                "annotation_id": a,
                "role": "alternate",
                "descriptor_id": "d",
                "value": w,
                "definedness": "defined" if np.isfinite(w) else "structural_absence",
            }
            for p, u, a, w in [
                ("p", "u", "a", 2),
                ("p", "u", "b", 4),
                ("p", "v", "a", nan),
                ("q", "w", "a", 9),
            ]
        ]
    )
    # This deliberately differs from a least-squares refit of the fixture.
    model = SourceModel(2, 3, 0.5, 0.25, 2, 20)
    result = source_residual_diagnostics(
        references,
        alternates,
        cohort="source",
        descriptor_ids=("d",),
        source_models={"d": model},
        families={"d": "f"},
    )
    eligible = result.loc[result.eligible]
    assert eligible.hierarchical_weight.tolist() == [0.25, 0.25, 0.5]
    assert np.allclose(eligible.source_projection_residual, [0.75, 2.75, 2.75])
    assert result.original_weighted_support.unique().tolist() == [0.75]
    assert not result.model_refitted.any()
    assert result.loc[~result.eligible, "hierarchical_weight"].tolist() == [0]
    assert result.loc[~result.eligible, "source_projection_residual"].isna().all()


def _fake_input_run(root, cfg, stage, outputs, *, extra=None, inputs=None):
    root.mkdir(parents=True)
    for name, value in outputs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, pd.DataFrame):
            value.to_csv(path, index=False)
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
    manifest = {
        "protocol_sha256": content_hash(cfg),
        "status": "complete",
        "stage": stage,
        "outputs": {name: file_hash(root / name) for name in outputs},
        "inputs": {str(path.resolve()): file_hash(path) for path in inputs or []},
        "extra": extra or {},
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_coverage_permission_requires_current_preparation_and_traceable_summary(
    tmp_path,
):
    cfg = config()
    prepared, other = tmp_path / "prepared", tmp_path / "other"
    _fake_input_run(prepared, cfg, "prepare_inputs", {})
    _fake_input_run(
        other, cfg, "prepare_inputs", {"identity.json": {"different": True}}
    )
    decision = {
        "nominal_audit_intervals_allowed": True,
        "required_cells_complete": True,
        "shortfall_count": 0,
        "source_table": "coverage_summary.csv",
    }
    summary = pd.DataFrame(
        [
            {
                "template": "fixture",
                "regime": "additive",
                "endpoint": "utility:first",
                "material_coverage_shortfall": False,
            }
        ]
    )
    coverage = tmp_path / "coverage"
    _fake_input_run(
        coverage,
        cfg,
        "simulation_coverage",
        {"coverage_decision.json": decision, "coverage_summary.csv": summary},
        inputs=[prepared / "run_manifest.json"],
    )
    paths = []
    assert _coverage_nominal_permission(coverage, cfg, prepared, paths) is True
    assert coverage / "coverage_summary.csv" in paths
    assert coverage / "coverage_decision.json" in paths
    assert coverage / "run_manifest.json" in paths
    with pytest.raises(ValueError, match="prepared input run"):
        _coverage_nominal_permission(coverage, cfg, other, [])


def test_coverage_boolean_strings_and_summary_disagreement_are_rejected(tmp_path):
    cfg = config()
    prepared = tmp_path / "prepared"
    _fake_input_run(prepared, cfg, "prepare_inputs", {})
    summary = pd.DataFrame([{"material_coverage_shortfall": False}])
    decision = {
        "nominal_audit_intervals_allowed": "false",
        "required_cells_complete": True,
        "shortfall_count": 0,
        "source_table": "coverage_summary.csv",
    }
    coverage = tmp_path / "boolean_string"
    _fake_input_run(
        coverage,
        cfg,
        "simulation_coverage",
        {"coverage_decision.json": decision, "coverage_summary.csv": summary},
        inputs=[prepared / "run_manifest.json"],
    )
    with pytest.raises(ValueError, match="JSON Booleans"):
        _coverage_nominal_permission(coverage, cfg, prepared, [])
    decision.update(nominal_audit_intervals_allowed=True)
    coverage = tmp_path / "wrong_summary"
    _fake_input_run(
        coverage,
        cfg,
        "simulation_coverage",
        {
            "coverage_decision.json": decision,
            "coverage_summary.csv": summary.assign(material_coverage_shortfall=True),
        },
        inputs=[prepared / "run_manifest.json"],
    )
    with pytest.raises(ValueError, match="disagrees"):
        _coverage_nominal_permission(coverage, cfg, prepared, [])


def test_file_evaluation_checks_committed_lineage_and_stamps_every_output(
    monkeypatch, tmp_path
):
    fixture, cfg = frames(), config(2)
    cfg.update(
        workspace={"descriptor_manifest": "metadata/descriptors.csv"},
        resolved_metadata={"patient_split": "metadata/split.csv"},
        cohorts={"target": {"modality": "mri"}},
    )
    (tmp_path / "metadata").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit/UNIT_CONTROL_SPECIFICATION.md").write_text(
        "Artificial fixture", encoding="utf-8"
    )
    (tmp_path / "config/protocol.lock.yaml").write_text(
        "artificial fixture", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"feature_id": descriptor, "domain": "mri", "family": family}
            for descriptor, family in fixture["descriptor_families"].items()
        ]
    ).to_csv(tmp_path / "metadata/descriptors.csv", index=False)
    pd.DataFrame(
        [
            {"cohort": "target", "patient_id": f"p{p:02d}", "assignment": "audit"}
            for p in range(12)
        ]
    ).to_csv(tmp_path / "metadata/split.csv", index=False)
    prepared, generation, construction = [
        tmp_path / name for name in ("prepared", "generation", "construction")
    ]
    _fake_input_run(
        prepared,
        cfg,
        "prepare_inputs",
        {
            "target/reference_features.csv": fixture["target_reference"],
            "target/alternate_features.csv": fixture["target_alternate"],
            "source/reference_features.csv": fixture["target_reference"].assign(
                cohort="source"
            ),
            "source/alternate_features.csv": fixture["target_alternate"].assign(
                cohort="source"
            ),
        },
    )
    _fake_input_run(
        generation,
        cfg,
        "generation",
        {
            "draw_features.csv": fixture["draw_features"],
            "draw_definedness.csv": fixture["draw_definedness"],
        },
        extra={"source": "source", "target": "target"},
    )
    prediction = {
        "reference_mean": 4,
        "reference_variance": 3,
        "disagreement_mean": 0.1,
        "disagreement_variance": 1,
        "reference_disagreement_covariance": 0,
        "disagreement_mse": 1.01,
        "ccc": 0.9,
        "reason": None,
    }
    _fake_input_run(
        construction,
        cfg,
        "construction",
        {
            "construction_status.json": {
                "universe": list(fixture["universe"]),
                "ready_for_commitment": True,
            },
            "construction_scores.csv": pd.DataFrame(
                [
                    {"policy": policy, "descriptor_id": descriptor, "score": score}
                    for policy, scores in fixture["score_maps"].items()
                    for descriptor, score in scores.items()
                ]
            ),
            "selections.csv": pd.DataFrame(
                [
                    {
                        "policy": screen.policy,
                        "descriptor_id": descriptor,
                        "rank": rank + 1,
                    }
                    for screen in fixture["selections"]
                    for rank, descriptor in enumerate(screen.selected_ids)
                ]
            ),
            "source_models.csv": pd.DataFrame(
                [
                    {
                        "descriptor_id": descriptor,
                        **asdict(model),
                        "source_disagreement_mean": 0.1,
                        "source_disagreement_variance": 1,
                        "source_reference_disagreement_covariance": 0,
                        "source_disagreement_mse": 1.01,
                        "source_ccc": 2 / 3.01,
                        "source_ccc_reason": None,
                    }
                    for descriptor, model in fixture["source_models"].items()
                ]
            ),
            "target_predictions.csv": pd.DataFrame(
                [
                    {"descriptor_id": descriptor, **prediction}
                    for descriptor in fixture["universe"]
                ]
            ),
            "synthetic_moments.csv": pd.DataFrame(
                [
                    {
                        "descriptor_id": descriptor,
                        **asdict(JointMoments(4, 3, 0.1, 1, 0, 1.01, 6 / 7.01)),
                    }
                    for descriptor in fixture["universe"]
                ]
            ),
        },
        extra={"direction": "source_to_target", "scope": "full"},
        inputs=[prepared / "run_manifest.json", generation / "run_manifest.json"],
    )
    commitment = {
        "direction": "source_to_target",
        "scope": "full",
        "universe": list(fixture["universe"]),
        "selections": {
            screen.policy: list(screen.selected_ids) for screen in fixture["selections"]
        },
        "artifacts": {
            str(construction / "run_manifest.json"): file_hash(
                construction / "run_manifest.json"
            )
        },
    }
    commitment_path = tmp_path / "commitment.json"
    commitment_path.write_text(json.dumps(commitment), encoding="utf-8")
    monkeypatch.setattr(
        "redesign2.evaluate.require_all_constructions", lambda *_: [commitment]
    )
    monkeypatch.setattr(
        "redesign2.prepare_inputs.require_protocol_lock", lambda *_: cfg
    )
    monkeypatch.setattr(
        "redesign2.provenance.code_state", lambda *_: {"artificial_test": True}
    )
    monkeypatch.setattr(
        "redesign2.provenance.environment", lambda: {"artificial_test": True}
    )
    result = evaluate_direction(
        tmp_path,
        cfg,
        prepared,
        generation,
        construction,
        [commitment_path],
        "source",
        "target",
        "full",
        "evaluation_fixture",
    )
    manifest = json.loads((result / "run_manifest.json").read_text(encoding="utf-8"))
    assert verify_run(result) == []
    assert len(manifest["outputs"]) == 22
    assert str(prepared / "source/alternate_features.csv") in manifest["inputs"]
    for name in manifest["outputs"]:
        table = pd.read_csv(result / name)
        assert {"protocol_sha256", "run_id", "declared_estimand"} <= set(table)
    # Changing a manifest while leaving the old commitment intact blocks before
    # reading any target alternate (the independently tested all-eight gate is mocked).
    committed_manifest = construction / "run_manifest.json"
    committed_manifest.write_text(
        committed_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="committed artifact"):
        evaluate_direction(
            tmp_path,
            cfg,
            prepared,
            generation,
            construction,
            [commitment_path],
            "source",
            "target",
            "full",
            "must_not_create",
        )
