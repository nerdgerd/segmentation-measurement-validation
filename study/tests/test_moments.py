"""Small joint distributions expose changes of estimand and sampling target."""

from dataclasses import astuple
from math import isnan

import numpy as np
import pytest

from redesign2.moments import (
    compare_disagreement,
    hierarchical_weights,
    pool_joint_moments,
    summarize_disagreement,
    weighted_joint_moments,
    weighted_reference_moments,
)


@pytest.mark.parametrize(
    "regime", ["additive", "bias", "proportional", "heteroscedastic", "negative_ccc"]
)
def test_general_identity_against_direct_ccc(regime):
    x = np.repeat([-2.0, -1.0, 1.0, 3.0], 2)
    error = np.tile([-0.5, 0.5], 4)
    if regime == "bias":
        error += 1.3
    elif regime == "proportional":
        error += -0.7 * x + 0.3
    elif regime == "heteroscedastic":
        error *= np.exp(0.3 * x)
    elif regime == "negative_ccc":
        error += -2.0 * x
    w = x + error
    weights = np.repeat([0.1, 0.2, 0.3, 0.4], 2) / 2
    moments = weighted_joint_moments(x, w, weights)
    xm, wm = weights @ x, weights @ w
    direct = (
        2
        * (weights @ ((x - xm) * (w - wm)))
        / (weights @ ((x - xm) ** 2) + weights @ ((w - wm) ** 2) + (xm - wm) ** 2)
    )
    assert moments.ccc == pytest.approx(direct, abs=2e-15)
    assert moments.disagreement_mse == pytest.approx(weights @ error**2)
    assert moments.disagreement_mse == pytest.approx(
        moments.disagreement_variance + moments.disagreement_mean**2
    )
    if regime == "negative_ccc":
        assert moments.ccc < 0
    if regime == "proportional":
        naive = (
            2
            * moments.reference_variance
            / (2 * moments.reference_variance + moments.disagreement_mse)
        )
        assert abs(moments.ccc - naive) > 0.1


def test_hierarchy_rebuilds_equal_patients_and_units_and_retains_original_coverage():
    support = hierarchical_weights(
        ["p1", "p1", "p2", "p2", "p2"],
        ["a", "a", "b", "c", "c"],
        ["r1", "r2", "r1", "r1", "r2"],
        eligible=[True, False, True, True, True],
    )
    assert support.weights == pytest.approx((0.5, 0, 0.25, 0.125, 0.125))
    assert support.original_support_coverage == pytest.approx(0.75)
    assert (
        support.eligible_patients,
        support.eligible_units,
        support.eligible_observations,
    ) == (2, 3, 4)
    assert sum(support.weights[:2]) == sum(support.weights[2:]) == 0.5
    with pytest.raises(ValueError, match="Repeated"):
        hierarchical_weights(
            ["p1", "p1"], ["a", "a"], ["draw1", "draw1"], eligible=[True, True]
        )


def test_patient_moment_pool_is_single_draw_mixture_not_ensemble():
    reference = [-1.0, 1.0]
    draws = [[-3.0, 1.0], [-1.0, 3.0]]
    groups = [
        weighted_joint_moments([x, x], values, [0.5, 0.5])
        for x, values in zip(reference, draws)
    ]
    pooled = pool_joint_moments(groups, [0.5, 0.5])
    expanded = weighted_joint_moments([-1, -1, 1, 1], [-3, 1, -1, 3], [0.25] * 4)
    for observed, expected in zip(astuple(pooled)[:-1], astuple(expanded)[:-1]):
        assert observed == pytest.approx(expected)
    ensemble = weighted_joint_moments(reference, np.mean(draws, axis=1), [0.5, 0.5])
    assert pooled.ccc == pytest.approx(1 / 3)
    assert ensemble.ccc == 1


@pytest.mark.parametrize("scale", [1e-8, 0.1, 23.0, 1e8, -4.0])
def test_units_offsets_and_oriented_bias(scale):
    x = np.array([-2.0, -1.0, 1.0, 3.0])
    observed = x + np.array([0.0, 0.2, -0.3, 1.1])
    synthetic = x + np.array([0.4, 0.4, -0.1, 0.7])
    weights = [0.1, 0.2, 0.3, 0.4]
    real = weighted_joint_moments(x, observed, weights)
    synth = weighted_joint_moments(x, synthetic, weights)
    shift = 31 * scale
    transformed_real = weighted_joint_moments(
        scale * x + shift, scale * observed + shift, weights
    )
    transformed_synth = weighted_joint_moments(
        scale * x + shift, scale * synthetic + shift, weights
    )
    before, after = (
        compare_disagreement(synth, real),
        compare_disagreement(transformed_synth, transformed_real),
    )
    assert transformed_real.ccc == pytest.approx(real.ccc, abs=3e-14)
    assert transformed_synth.ccc == pytest.approx(synth.ccc, abs=3e-14)
    assert after.log2_rmse_ratio == pytest.approx(before.log2_rmse_ratio, abs=1e-13)
    assert after.reference_slope_difference == pytest.approx(
        before.reference_slope_difference, abs=1e-14
    )
    assert after.standardized_bias_difference == pytest.approx(
        np.sign(scale) * before.standardized_bias_difference, abs=1e-13
    )


def test_zero_discrepancies_are_distinct_and_finite_summary_has_denominator():
    exact = weighted_joint_moments([-1, 1], [-1, 1], [0.5, 0.5])
    error = weighted_joint_moments([-1, 1], [0, 2], [0.5, 0.5])
    comparisons = [
        compare_disagreement(a, b)
        for a, b in ((exact, exact), (exact, error), (error, exact), (error, error))
    ]
    assert [item.rmse_stratum for item in comparisons] == [
        "both_zero",
        "synthetic_only_zero",
        "observed_only_zero",
        "finite",
    ]
    assert isnan(comparisons[0].log2_rmse_ratio)
    assert comparisons[1].log2_rmse_ratio == -np.inf
    assert comparisons[2].log2_rmse_ratio == np.inf
    summary = summarize_disagreement(comparisons)
    assert summary["n_total"] == 4 and summary["n_finite"] == 1
    assert summary["median_signed_finite_log2_ratio"] == 0
    assert all(
        summary["n_" + label] == 1
        for label in ("both_zero", "synthetic_only_zero", "observed_only_zero")
    )


def test_missingness_is_never_implicitly_dropped_and_undefined_is_not_zero():
    with pytest.raises(ValueError, match="eligible support"):
        weighted_joint_moments([1, 2], [1, np.nan], [0.5, 0.5])
    with pytest.raises(ValueError, match="normalized"):
        weighted_reference_moments([1, 2], [1, 1])
    same_constant = weighted_joint_moments([2, 2], [2, 2], [0.5, 0.5])
    assert isnan(same_constant.ccc)
    assert same_constant.ccc_reason == "nonpositive_ccc_denominator"


def test_reference_variance_uses_centered_offsets_not_difference_of_large_squares():
    base = np.array([-2.0, -1.0, 1.0, 2.0])
    result = weighted_reference_moments(base + 1e12, [0.25] * 4)
    assert result.variance == 2.5
