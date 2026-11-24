"""Recovery under a fixed conditional law and recorded failures are both gates."""

from dataclasses import replace

import numpy as np
import pytest

from redesign2.moments import (
    ReferenceMoments,
    weighted_joint_moments,
    weighted_reference_moments,
)
from redesign2.transport import (
    fit_source_model,
    predict_target,
    target_extrapolation_fraction,
)


def known_law(reference, *, intercept=0.4, slope=-0.3, residual=0.8):
    x = np.repeat(np.asarray(reference, dtype=float), 2)
    noise = np.tile([-residual, residual], len(reference))
    w = x + intercept + slope * x + noise
    weights = np.full(len(x), 1 / len(x))
    return x, w, weights


@pytest.mark.parametrize("variance_multiplier", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("mean_shift", [0.0, 1.0])
def test_exact_source_and_target_recovery_under_unchanged_linear_law(
    variance_multiplier, mean_shift
):
    base = np.array([-2, -1, 0, 1, 2], dtype=float)
    x, w, weights = known_law(base)
    source = weighted_joint_moments(x, w, weights)
    model = fit_source_model(source, independent_patients=10)
    assert model.slope == pytest.approx(-0.3)
    assert model.mean_disagreement == pytest.approx(0.4)
    assert model.residual_variance == pytest.approx(0.8**2)
    reconstructed_source = predict_target(model, weighted_reference_moments(x, weights))
    assert reconstructed_source.ccc == pytest.approx(source.ccc)
    xt, wt, target_weights = known_law(
        np.sqrt(variance_multiplier) * base + mean_shift * np.std(base)
    )
    prediction = predict_target(model, weighted_reference_moments(xt, target_weights))
    actual = weighted_joint_moments(xt, wt, target_weights)
    for field in (
        "ccc",
        "disagreement_mean",
        "disagreement_variance",
        "reference_disagreement_covariance",
        "disagreement_mse",
    ):
        assert getattr(prediction, field) == pytest.approx(
            getattr(actual, field), abs=2e-15
        )


@pytest.mark.parametrize("scale", [1e-8, 0.01, 10, 1e8, -3])
def test_transport_affine_unit_invariance_without_absolute_variance_floor(scale):
    x, w, weights = known_law([-2, -1, 0, 1, 2])
    target = 1.4 * x + 2
    original_model = fit_source_model(
        weighted_joint_moments(x, w, weights), independent_patients=10
    )
    original = predict_target(
        original_model, weighted_reference_moments(target, weights)
    )
    shift = 19 * scale
    converted_model = fit_source_model(
        weighted_joint_moments(scale * x + shift, scale * w + shift, weights),
        independent_patients=10,
    )
    converted = predict_target(
        converted_model, weighted_reference_moments(scale * target + shift, weights)
    )
    assert converted.defined
    assert converted.ccc == pytest.approx(original.ccc, abs=1e-14)
    assert converted_model.slope == pytest.approx(original_model.slope, abs=1e-14)
    assert converted.disagreement_mse == pytest.approx(
        scale**2 * original.disagreement_mse
    )


def test_valid_negative_transport_ccc_is_retained():
    x, w, weights = known_law([-2, -1, 0, 1, 2], slope=-2)
    source = weighted_joint_moments(x, w, weights)
    prediction = predict_target(
        fit_source_model(source, independent_patients=10), ReferenceMoments(0, 2)
    )
    assert prediction.ccc == pytest.approx(source.ccc)
    assert prediction.ccc < 0


def test_roundoff_clamp_is_relative_logged_and_large_invalid_moment_is_error():
    x, w, weights = known_law([-2, -1, 0, 1, 2], residual=0, slope=0.5)
    source = weighted_joint_moments(x, w, weights)
    tiny = replace(
        source, disagreement_variance=source.disagreement_variance * (1 - 1e-12)
    )
    model = fit_source_model(tiny, independent_patients=10)
    assert model.residual_variance == 0
    assert model.residual_variance_roundoff_clamped
    assert model.residual_variance_before_clamp < 0
    bad = replace(source, disagreement_variance=source.disagreement_variance * 0.99)
    with pytest.raises(ValueError, match="exceeds"):
        fit_source_model(bad, independent_patients=10)


def test_degenerate_references_and_insufficient_source_count_are_not_ranked():
    constant = weighted_joint_moments([2, 2], [2, 3], [0.5, 0.5])
    model = fit_source_model(constant, independent_patients=10)
    assert not model.defined
    assert model.reason == "nonpositive_source_reference_variance"
    x, w, weights = known_law([-2, -1, 0, 1, 2])
    source = weighted_joint_moments(x, w, weights)
    assert (
        fit_source_model(source, independent_patients=9).reason
        == "insufficient_independent_source_patients"
    )
    target = predict_target(
        fit_source_model(source, independent_patients=10), ReferenceMoments(1, 0)
    )
    assert target.reason == "constant_target_reference"
    assert target.disagreement_mse > 0
    assert np.isnan(target.ccc)


def test_heteroscedastic_case_mix_shift_exposes_the_frozen_model_failure():
    x = np.repeat([-2, -1, 0, 1, 2], 2).astype(float)
    signs = np.tile([-1, 1], 5)
    w = x + 0.4 - 0.3 * x + 0.8 * np.exp(0.3 * x) * signs
    weights = [0.1] * 10
    model = fit_source_model(
        weighted_joint_moments(x, w, weights), independent_patients=10
    )
    xt = x + 2
    wt = xt + 0.4 - 0.3 * xt + 0.8 * np.exp(0.3 * xt) * signs
    predicted = predict_target(model, weighted_reference_moments(xt, weights))
    observed = weighted_joint_moments(xt, wt, weights)
    assert abs(predicted.ccc - observed.ccc) > 0.1


def test_weighted_support_extrapolation_is_only_a_diagnostic():
    result = target_extrapolation_fraction(
        [0, 1, 2, 3], [0.005, 0.495, 0.495, 0.005], [0, 1, 2, 4], [0.1, 0.2, 0.3, 0.4]
    )
    assert result == {
        "source_lower_quantile": 1,
        "source_upper_quantile": 2,
        "target_weighted_fraction_outside": 0.5,
    }
