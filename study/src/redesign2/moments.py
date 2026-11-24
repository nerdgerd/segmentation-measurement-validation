"""Integrate random contour measurements without converting them to ensembles.

Eligibility is supplied by audited adapters, never inferred from numeric values.
Weights describe the sampling target; all moments use their population form.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite, log2, nan, sqrt
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SupportWeights:
    weights: tuple[float, ...]
    original_support_coverage: float
    eligible_patients: int
    eligible_units: int
    eligible_observations: int


@dataclass(frozen=True)
class ReferenceMoments:
    mean: float
    variance: float


@dataclass(frozen=True)
class JointMoments:
    reference_mean: float
    reference_variance: float
    disagreement_mean: float
    disagreement_variance: float
    reference_disagreement_covariance: float
    disagreement_mse: float
    ccc: float
    ccc_reason: str | None = None


@dataclass(frozen=True)
class DisagreementComparison:
    log2_rmse_ratio: float
    rmse_stratum: str
    standardized_bias_difference: float
    synthetic_reference_slope: float
    observed_reference_slope: float
    reference_slope_difference: float


def hierarchical_weights(
    patient_ids: Sequence[str],
    unit_ids: Sequence[str],
    observation_ids: Sequence[str],
    *,
    eligible: Sequence[bool],
) -> SupportWeights:
    """Give patients, their eligible units, and their observations equal weight.

    All rows describe the original audited support, including explicitly ineligible
    rows. Coverage uses that original support. Conditional weights are rebuilt at
    each level after eligibility is applied; merely renormalizing retained row
    weights would give patients with missing readers unequal weight.
    """
    n = len(patient_ids)
    if not (len(unit_ids) == len(observation_ids) == len(eligible) == n):
        raise ValueError("Support arrays must have equal lengths")
    if any(not isinstance(flag, (bool, np.bool_)) for flag in eligible):
        raise TypeError("Eligibility must be an explicit Boolean mask")
    keys = list(zip(patient_ids, unit_ids, observation_ids))
    if any(not isinstance(v, str) or not v for key in keys for v in key):
        raise ValueError("Support identifiers must be nonempty strings")
    if len(set(keys)) != n:
        raise ValueError("Repeated patient/unit/observation identity")

    def weights_for(indices: list[int]) -> tuple[np.ndarray, int, int]:
        units = defaultdict(set)
        counts = Counter()
        for index in indices:
            patient, unit, _ = keys[index]
            units[patient].add(unit)
            counts[patient, unit] += 1
        values = np.zeros(n, dtype=float)
        for index in indices:
            patient, unit, _ = keys[index]
            values[index] = 1.0 / (
                len(units) * len(units[patient]) * counts[patient, unit]
            )
        return values, len(units), len(counts)

    original, _, _ = weights_for(list(range(n)))
    kept = [i for i, flag in enumerate(eligible) if flag]
    conditional, patients, units = weights_for(kept)
    return SupportWeights(
        tuple(conditional), float(original[kept].sum()), patients, units, len(kept)
    )


def normalized_weights(weights: Sequence[float], size: int) -> np.ndarray:
    """Validate already normalized support, allowing floating-point sum drift."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != size or size == 0:
        raise ValueError(
            "A nonempty, aligned one-dimensional weight vector is required"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Weights must be finite and nonnegative")
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=1e-12, atol=0):
        raise ValueError("Weights must describe explicitly normalized eligible support")
    return values / total


def _numeric_values(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(
            "Values must be finite and one-dimensional; select eligible support explicitly"
        )
    return result


def _center(values: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    offset = values - values[0]
    mean_offset = float(np.dot(weights, offset))
    return float(values[0] + mean_offset), offset - mean_offset


def weighted_reference_moments(
    reference: Sequence[float], weights: Sequence[float]
) -> ReferenceMoments:
    values = _numeric_values(reference)
    normalized = normalized_weights(weights, len(values))
    mean, centered = _center(values, normalized)
    return ReferenceMoments(mean, float(np.dot(normalized, centered * centered)))


def _joint_from_components(
    mu: float, v: float, b: float, q: float, a: float
) -> JointMoments:
    disagreement_mse = q + b * b
    numerator = 2 * (v + a)
    denominator = numerator + disagreement_mse
    required = (mu, v, b, q, a, disagreement_mse, denominator)
    if not all(isfinite(value) for value in required):
        ccc, reason = nan, "nonfinite_required_moment"
    elif denominator <= 0:
        ccc, reason = nan, "nonpositive_ccc_denominator"
    else:
        ccc, reason = numerator / denominator, None
    return JointMoments(mu, v, b, q, a, disagreement_mse, ccc, reason)


def weighted_joint_moments(
    reference: Sequence[float], alternate: Sequence[float], weights: Sequence[float]
) -> JointMoments:
    """CCC of one random alternate from the supplied weighted joint support."""
    x, w = _numeric_values(reference), _numeric_values(alternate)
    if len(x) != len(w):
        raise ValueError("Reference and alternate values must be aligned")
    normalized = normalized_weights(weights, len(x))
    mu, dx = _center(x, normalized)
    b, dd = _center(w - x, normalized)
    return _joint_from_components(
        mu,
        float(np.dot(normalized, dx * dx)),
        b,
        float(np.dot(normalized, dd * dd)),
        float(np.dot(normalized, dx * dd)),
    )


def pool_joint_moments(
    groups: Sequence[JointMoments], weights: Sequence[float]
) -> JointMoments:
    """Pool conditional moments, retaining within-group measurement variance."""
    normalized = normalized_weights(weights, len(groups))
    means = _numeric_values([group.reference_mean for group in groups])
    biases = _numeric_values([group.disagreement_mean for group in groups])
    mu, dm = _center(means, normalized)
    b, db = _center(biases, normalized)
    v = _numeric_values([group.reference_variance for group in groups])
    q = _numeric_values([group.disagreement_variance for group in groups])
    a = _numeric_values([group.reference_disagreement_covariance for group in groups])
    return _joint_from_components(
        mu,
        float(np.dot(normalized, v + dm * dm)),
        b,
        float(np.dot(normalized, q + db * db)),
        float(np.dot(normalized, a + dm * db)),
    )


def measurement_difference_ratio(ds: float, dr: float) -> tuple[float, str]:
    """Return log₂(generated RMSE / original RMSE), keeping zero cases explicit."""
    if not all(isfinite(v) and v >= 0 for v in (ds, dr)):
        ratio, stratum = nan, "undefined"
    elif ds == dr == 0:
        ratio, stratum = nan, "both_zero"
    elif ds == 0:
        ratio, stratum = -float("inf"), "synthetic_only_zero"
    elif dr == 0:
        ratio, stratum = float("inf"), "observed_only_zero"
    else:
        ratio, stratum = 0.5 * (log2(ds) - log2(dr)), "finite"
    return ratio, stratum


def compare_disagreement(
    synthetic: JointMoments, observed: JointMoments
) -> DisagreementComparison:
    """Keep exact zero strata; signed bias retains descriptor orientation."""
    ds, dr = synthetic.disagreement_mse, observed.disagreement_mse
    ratio, stratum = measurement_difference_ratio(ds, dr)
    bias = (
        (synthetic.disagreement_mean - observed.disagreement_mean) / sqrt(dr)
        if isfinite(dr) and dr > 0
        else nan
    )
    slopes = [
        group.reference_disagreement_covariance / group.reference_variance
        if isfinite(group.reference_variance) and group.reference_variance > 0
        else nan
        for group in (synthetic, observed)
    ]
    return DisagreementComparison(
        ratio, stratum, bias, slopes[0], slopes[1], slopes[0] - slopes[1]
    )


def summarize_disagreement(
    comparisons: Sequence[DisagreementComparison],
) -> dict[str, float | int]:
    finite = [
        item.log2_rmse_ratio for item in comparisons if item.rmse_stratum == "finite"
    ]
    counts = Counter(item.rmse_stratum for item in comparisons)
    return {
        "n_total": len(comparisons),
        "n_finite": len(finite),
        "median_signed_finite_log2_ratio": float(np.median(finite)) if finite else nan,
        "median_absolute_finite_log2_ratio": float(np.median(np.abs(finite)))
        if finite
        else nan,
        **{
            f"n_{label}": counts[label]
            for label in (
                "both_zero",
                "synthetic_only_zero",
                "observed_only_zero",
                "undefined",
            )
        },
    }
