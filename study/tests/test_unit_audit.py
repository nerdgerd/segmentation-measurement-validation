"""Whole-battery and direct-value controls share unchanged declared support."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from redesign2.construction import MatrixMeasurements
from redesign2.empirical_math import aggregate_patient_moments
from redesign2.moments import JointMoments, ReferenceMoments
from redesign2.provenance import read_protocol
from redesign2 import unit_audit
from redesign2.unit_audit import (
    _decimal_equal,
    _precise_slope_control,
    audit_unit_conversions,
    descriptor_transform,
)


def unit_inputs():
    cfg = read_protocol(Path(__file__).resolve().parents[1] / "config/protocol.yaml")
    ids = tuple(f"d{index:02}" for index in range(10))
    patient = np.arange(12, dtype=float)[:, None]
    feature = np.arange(10, dtype=float)[None, :]
    x = (patient - 5) * (1 + feature * 0.1) + feature
    source_w = x + 0.04 * feature + np.sin(patient) * (0.1 + feature * 0.03)
    target_x = 1.3 * x + 1
    synthetic_w = target_x + (0.5 - feature * 0.02) * np.cos(patient)
    observed_w = target_x + 0.04 * feature + (0.1 + feature * 0.04) * np.sin(patient)
    keys = pd.DataFrame(
        {
            "patient_id": [f"p{index:02}" for index in range(12)],
            "unit_id": "unit",
            "observation_id": "measurement",
        }
    )
    eligible = np.ones_like(x, dtype=bool)
    matrices = {
        "source": MatrixMeasurements(ids, keys, x, source_w, eligible),
        "target_reference": MatrixMeasurements(
            ids, keys, target_x, target_x.copy(), eligible
        ),
        "synthetic": MatrixMeasurements(ids, keys, target_x, synthetic_w, eligible),
        "observed": MatrixMeasurements(ids, keys, target_x, observed_w, eligible),
    }
    profiles = {
        role: aggregate_patient_moments(matrix.patient_moments())
        for role, matrix in matrices.items()
    }
    source = dict(zip(ids, profiles["source"].moments))
    synthetic = dict(zip(ids, profiles["synthetic"].moments))
    observed = dict(zip(ids, profiles["observed"].moments))
    references = {
        descriptor: ReferenceMoments(moment.reference_mean, moment.reference_variance)
        for descriptor, moment in zip(ids, profiles["target_reference"].moments)
    }
    arguments = {
        "direction": "upenn_to_brats_tcga",
        "scope": "full",
        "source_moments": source,
        "target_reference": references,
        "synthetic_moments": synthetic,
        "observed_moments": observed,
        "diagnostic_synthetic_moments": synthetic.copy(),
        "source_patient_counts": dict.fromkeys(ids, 12),
        "matrices": matrices,
    }
    return cfg, arguments


def test_complete_empirical_style_unit_audit_preserves_primary_scores_and_selection():
    cfg, arguments = unit_inputs()
    result = audit_unit_conversions(cfg, **arguments)
    gates = result.loc[result.passed.notna()]
    assert gates.passed.all()
    scores = result.loc[result.audit_component.eq("whole_battery_policy_score")]
    assert len(scores) == 30
    selections = result.loc[result.audit_component.eq("whole_battery_policy_selection")]
    assert len(selections) == 3
    assert (selections.selected_before == selections.selected_after).all()
    direct = result.loc[
        result.audit_component.eq("direct_empirical_values_first_eight_ids")
    ]
    assert len(direct) == 32
    assert set(direct.descriptor_id) == {f"d{index:02}" for index in range(8)}
    raw_iqr = result.loc[
        result.audit_component.eq("raw_weighted_reference_iqr_unit_sensitivity")
    ]
    assert len(raw_iqr) == 10
    assert set(raw_iqr.rank_before) == set(range(1, 11))
    assert set(raw_iqr.rank_after) == set(range(1, 11))
    assert raw_iqr.passed.isna().all()


def test_control_choice_uses_identifiers_and_source_scale_not_target_performance():
    cfg, arguments = unit_inputs()
    first = audit_unit_conversions(cfg, **arguments)
    arguments["observed_moments"] = {
        descriptor: replace(
            moment,
            disagreement_mean=moment.disagreement_mean + 2,
            disagreement_mse=moment.disagreement_variance
            + (moment.disagreement_mean + 2) ** 2,
        )
        for descriptor, moment in arguments["observed_moments"].items()
    }
    second = audit_unit_conversions(cfg, **arguments)
    before = first.loc[
        first.audit_component.eq("whole_battery_policy_score"),
        ["descriptor_id", "input_role", "scale", "shift", "transform_key_sha256"],
    ]
    after = second.loc[
        second.audit_component.eq("whole_battery_policy_score"), before.columns
    ]
    pd.testing.assert_frame_equal(before, after)
    scale_a = descriptor_transform(cfg, "d00", 1)
    scale_b = descriptor_transform(cfg, "d00", 3)
    assert scale_b["scale"] == scale_a["scale"]
    assert scale_b["shift"] == 3 * scale_a["shift"]


def test_disagreement_uses_separate_matched_synthetic_moments_and_retains_zero_strata():
    cfg, arguments = unit_inputs()
    arguments["matrices"] = None
    for descriptor in ("d00", "d01", "d02"):
        observed = arguments["observed_moments"][descriptor]
        exact = replace(
            observed,
            disagreement_mean=0,
            disagreement_variance=0,
            reference_disagreement_covariance=0,
            disagreement_mse=0,
            ccc=1,
        )
        if descriptor in ("d00", "d01"):
            arguments["observed_moments"][descriptor] = exact
        if descriptor in ("d01", "d02"):
            arguments["diagnostic_synthetic_moments"][descriptor] = exact
    result = audit_unit_conversions(cfg, **arguments)
    ratios = result.loc[result.input_role.eq("log2_rmse_ratio")].set_index(
        "descriptor_id"
    )
    assert ratios.loc["d00", "rmse_stratum_before"] == "observed_only_zero"
    assert ratios.loc["d01", "rmse_stratum_before"] == "both_zero"
    assert ratios.loc["d02", "rmse_stratum_before"] == "synthetic_only_zero"
    assert ratios.passed.all()
    assert result.loc[
        result.audit_component.eq("whole_battery_policy_score"), "passed"
    ].all()
    assert "not_run_no_matrices_supplied" in result.status.to_numpy()


def test_audit_does_not_intersect_away_a_missing_locked_construction_descriptor():
    cfg, arguments = unit_inputs()
    del arguments["synthetic_moments"]["d00"]
    with pytest.raises(ValueError, match="share the locked"):
        audit_unit_conversions(cfg, **arguments)


def large_slope_moments():
    # Stored-moment reproducer for the failure documented on 2026-09-16.
    observed = JointMoments(
        0.8404754471847244,
        2.535993365354645,
        693877.3852441057,
        70294038421455.46,
        2395249.257958218,
        70775504247208.66,
        6.76858937934389e-08,
    )
    synthetic = JointMoments(
        0.8404754471847244,
        2.535993365354645,
        686122.4794185736,
        6156284722237742.0,
        11378118.68909368,
        6156755486294505.0,
        3.6961419722971367e-09,
    )
    return synthetic, observed


def test_million_scale_slope_uses_precise_gate_and_retains_native_roundoff(monkeypatch):
    cfg, arguments = unit_inputs()
    synthetic, observed = large_slope_moments()
    arguments["diagnostic_synthetic_moments"]["d00"] = synthetic
    arguments["observed_moments"]["d00"] = observed
    original_transform = descriptor_transform

    def fixed_scale(cfg, descriptor, source_sd):
        result = original_transform(cfg, descriptor, source_sd)
        if descriptor == "d00":
            result.update(scale=-10.0, shift=0.0)
        return result

    monkeypatch.setattr(unit_audit, "descriptor_transform", fixed_scale)
    result = audit_unit_conversions(cfg, **arguments)
    row = result.loc[
        result.descriptor_id.eq("d00")
        & result.input_role.eq("reference_slope_difference")
    ].iloc[0]
    assert row.native_binary64_absolute_change == 9.313225746154785e-10
    assert row.native_binary64_absolute_change > row.dimensionless_tolerance
    assert row.passed
    assert row.absolute_change <= row.dimensionless_tolerance == 1e-10
    assert row.verification_arithmetic == "decimal80_exact_binary64_inputs"
    assert Decimal(row.value_before_precision_text) > 3_000_000
    assert Decimal(row.absolute_change_precision_text) <= Decimal("1e-10")
    assert result.loc[result.passed.notna(), "passed"].all()


@pytest.mark.parametrize("scale", [-1e6, -10.0, -1e-6, 1e-6, 10.0, 1e6])
def test_large_slope_precision_is_uniform_across_unit_scales(scale):
    synthetic, observed = large_slope_moments()
    result = _precise_slope_control(synthetic, observed, scale, scale * 3.4)
    assert result["passed"]
    assert result["absolute_change"] <= 1e-10


def test_precise_comparison_detects_difference_hidden_by_display_rounding():
    before, after = Decimal("3542150.2097972776"), Decimal("3542150.2097972778")
    assert float(before) == float(after)
    passed, difference = _decimal_equal(before, after)
    assert not passed
    assert difference == Decimal("2e-10")


def test_precise_slope_control_still_detects_real_transformation_defect(monkeypatch):
    original_transform = unit_audit.affine_joint_moments

    def broken_transform(moments, scale, shift):
        converted = original_transform(moments, scale, shift)
        if isinstance(moments.reference_variance, Decimal):
            return replace(
                converted,
                reference_disagreement_covariance=converted.reference_disagreement_covariance
                * Decimal("1.000001"),
            )
        return converted

    monkeypatch.setattr(unit_audit, "affine_joint_moments", broken_transform)
    cfg, arguments = unit_inputs()
    result = audit_unit_conversions(cfg, **arguments)
    slopes = result.loc[result.input_role.eq("reference_slope_difference")]
    assert (~slopes.passed.astype(bool)).any()
    assert (slopes.absolute_change > 1e-10).any()
    assert result.loc[
        result.audit_component.eq("whole_battery_policy_score"), "passed"
    ].all()


@pytest.mark.parametrize("variance", [0.0, float("nan"), float("inf"), float("-inf")])
def test_precise_slope_control_preserves_undefined_reference_slope(variance):
    synthetic, observed = large_slope_moments()
    result = _precise_slope_control(
        synthetic, replace(observed, reference_variance=variance), -10.0, 0.0
    )
    assert result["passed"]
    assert np.isnan(result["value_before"])
    assert np.isnan(result["value_after"])
