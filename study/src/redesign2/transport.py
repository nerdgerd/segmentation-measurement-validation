"""One source linear projection, with target reference moments as its only adaptation.

The residual second moment is transported unchanged. A source projection fit
does not establish that this conditional law holds in another cohort.
"""

from dataclasses import dataclass
from math import isfinite, nan
from typing import Sequence

import numpy as np

from .moments import JointMoments, ReferenceMoments, normalized_weights


@dataclass(frozen=True)
class SourceModel:
    source_mean: float
    source_reference_variance: float
    mean_disagreement: float
    slope: float
    residual_variance: float
    independent_patients: int
    reason: str | None = None
    residual_variance_roundoff_clamped: bool = False
    residual_variance_before_clamp: float = nan

    @property
    def defined(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class TransportPrediction:
    reference_mean: float
    reference_variance: float
    disagreement_mean: float
    disagreement_variance: float
    reference_disagreement_covariance: float
    disagreement_mse: float
    ccc: float
    reason: str | None = None

    @property
    def defined(self) -> bool:
        return self.reason is None


def fit_source_model(
    moments: JointMoments,
    *,
    independent_patients: int,
    min_independent_patients: int = 10,
    residual_roundoff_relative_tolerance: float = 1e-10,
) -> SourceModel:
    """Fit the specified WLS projection; support counts are adapter declarations."""
    mu, v = moments.reference_mean, moments.reference_variance
    b, q, a = (
        moments.disagreement_mean,
        moments.disagreement_variance,
        moments.reference_disagreement_covariance,
    )
    if (
        not isinstance(independent_patients, (int, np.integer))
        or isinstance(independent_patients, bool)
        or independent_patients < 0
    ):
        raise ValueError("Independent patient count must be a nonnegative integer")
    if min_independent_patients < 1 or residual_roundoff_relative_tolerance < 0:
        raise ValueError("Invalid predeclared numerical/support rule")
    reason = None
    if independent_patients < min_independent_patients:
        reason = "insufficient_independent_source_patients"
    elif not all(isfinite(value) for value in (mu, v, b, q, a)):
        reason = "nonfinite_source_moment"
    elif v <= 0:
        reason = "nonpositive_source_reference_variance"
    elif q < 0:
        raise ValueError("Negative disagreement variance is a numerical/data error")
    if reason:
        return SourceModel(mu, v, b, nan, nan, independent_patients, reason)
    slope = a / v
    projection_variance = slope * a
    residual = q - projection_variance
    if not all(isfinite(value) for value in (slope, residual)):
        return SourceModel(
            mu, v, b, slope, nan, independent_patients, "nonfinite_source_projection"
        )
    moment_scale = max(abs(q), abs(projection_variance))
    clamped = residual < 0
    if residual < -residual_roundoff_relative_tolerance * moment_scale:
        raise ValueError(
            "Negative residual variance exceeds the declared relative roundoff tolerance"
        )
    return SourceModel(
        mu,
        v,
        b,
        slope,
        max(0.0, residual),
        independent_patients,
        residual_variance_roundoff_clamped=clamped,
        residual_variance_before_clamp=residual,
    )


def predict_target(
    model: SourceModel, target_reference: ReferenceMoments
) -> TransportPrediction:
    """Reconstruct disagreement and CCC using target references only."""
    mu, v = target_reference.mean, target_reference.variance
    if not model.defined:
        return TransportPrediction(
            mu,
            v,
            nan,
            nan,
            nan,
            nan,
            nan,
            "undefined_source_model:" + str(model.reason),
        )
    if not all(isfinite(value) for value in (mu, v)) or v < 0:
        return TransportPrediction(
            mu, v, nan, nan, nan, nan, nan, "invalid_target_reference_moment"
        )
    b = model.mean_disagreement + model.slope * (mu - model.source_mean)
    a = model.slope * v
    q = model.slope * model.slope * v + model.residual_variance
    disagreement_mse = q + b * b
    denominator = (1 + (1 + model.slope) ** 2) * v + model.residual_variance + b * b
    reason = None
    if not all(isfinite(value) for value in (b, a, q, disagreement_mse, denominator)):
        reason = "nonfinite_prediction_moment"
    elif v == 0:
        reason = "constant_target_reference"
    elif denominator <= 0:
        reason = "nonpositive_ccc_denominator"
    ccc = 2 * (1 + model.slope) * v / denominator if reason is None else nan
    return TransportPrediction(mu, v, b, q, a, disagreement_mse, ccc, reason)


def weighted_quantile(
    values: Sequence[float], weights: Sequence[float], probability: float
) -> float:
    """Inverse weighted empirical CDF; interpolation never changes eligibility."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or not 0 <= probability <= 1:
        raise ValueError("Finite values and a probability in [0, 1] are required")
    normalized = normalized_weights(weights, len(values))
    positive = normalized > 0
    values, normalized = values[positive], normalized[positive]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(normalized[order])
    position = min(
        int(np.searchsorted(cumulative, probability, side="left")), len(order) - 1
    )
    return float(values[order[position]])


def target_extrapolation_fraction(
    source_reference: Sequence[float],
    source_weights: Sequence[float],
    target_reference: Sequence[float],
    target_weights: Sequence[float],
    *,
    quantiles: tuple[float, float] = (0.01, 0.99),
) -> dict[str, float]:
    if not 0 <= quantiles[0] <= quantiles[1] <= 1:
        raise ValueError("Support quantiles must be ordered probabilities")
    lower = weighted_quantile(source_reference, source_weights, quantiles[0])
    upper = weighted_quantile(source_reference, source_weights, quantiles[1])
    target = np.asarray(target_reference, dtype=float)
    if target.ndim != 1 or not np.all(np.isfinite(target)):
        raise ValueError("Target references must be explicitly eligible finite values")
    weights = normalized_weights(target_weights, len(target))
    outside = (target < lower) | (target > upper)
    return {
        "source_lower_quantile": lower,
        "source_upper_quantile": upper,
        "target_weighted_fraction_outside": float(weights[outside].sum()),
    }
