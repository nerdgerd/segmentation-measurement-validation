"""Patient bootstrap distributions with the conditioning stated in the result.

Fixed-audit screens remain fixed. Full-cohort adaptation is an explicit callback
that recomputes its rankings from each shared target-patient sample. Source fits,
severity, and the generated pool are never resampled by these primitives.
"""

from dataclasses import dataclass
from math import nan
from typing import Callable, Mapping, Sequence

import numpy as np

from .empirical_math import PatientMomentTable, aggregate_patient_moments
from .policies import PolicySelection, evaluate_selection, utility_contrasts


FULL_COHORT_CONDITIONING = "target_sampling_conditional_on_source_and_generated_pool"
AUDIT_CONDITIONING = "criterion_sampling_conditional_on_constructed_screen"


@dataclass(frozen=True)
class BootstrapDistribution:
    statistic_ids: tuple[str, ...]
    values: np.ndarray
    conditioning: str
    patient_count: int
    replicates: int


def patient_bootstrap_multiplicities(
    patient_count: int, replicates: int, *, rng: np.random.Generator
) -> np.ndarray:
    """Sample patient slots uniformly, retaining each selected patient's cluster."""
    if patient_count < 1 or replicates < 1:
        raise ValueError("Positive patient and replicate counts are required")
    draws = rng.integers(0, patient_count, size=(replicates, patient_count))
    counts = np.zeros((replicates, patient_count), dtype=int)
    np.add.at(
        counts, (np.repeat(np.arange(replicates), patient_count), draws.ravel()), 1
    )
    return counts


def conditional_bootstrap(
    *,
    patient_count: int,
    statistic_ids: Sequence[str],
    statistic: Callable[[np.ndarray], Mapping[str, float]],
    rng: np.random.Generator,
    conditioning: str,
    replicates: int = 2000,
) -> BootstrapDistribution:
    """One target sample is shared by every statistic and procedure per replicate.

    The callback returns undefined estimates as NaN with any detailed reason retained
    by its caller. Exceptions indicate technical errors and are not suppressed.
    """
    if conditioning not in (FULL_COHORT_CONDITIONING, AUDIT_CONDITIONING):
        raise ValueError("Unknown bootstrap conditioning")
    labels = tuple(statistic_ids)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError(
            "Unique predeclared bootstrap statistic identities are required"
        )
    values = np.empty((replicates, len(labels)), dtype=float)
    for index, counts in enumerate(
        patient_bootstrap_multiplicities(patient_count, replicates, rng=rng)
    ):
        result = statistic(counts)
        if set(result) != set(labels):
            raise ValueError("Bootstrap callback changed the statistic identities")
        values[index] = [result[label] for label in labels]
    values.setflags(write=False)
    return BootstrapDistribution(
        labels, values, conditioning, patient_count, replicates
    )


def fixed_selection_bootstrap(
    table: PatientMomentTable,
    selections: Sequence[PolicySelection],
    *,
    rng: np.random.Generator,
    replicates: int = 2000,
    min_independent_patients: int = 10,
    min_weighted_support: float = 0.90,
) -> BootstrapDistribution:
    """Audit criterion uncertainty conditional on the original constructed screen."""
    if not selections:
        raise ValueError("Fixed selected sets are required")
    universe = selections[0].universe
    if any(selection.universe != universe for selection in selections) or not set(
        universe
    ).issubset(table.descriptor_ids):
        raise ValueError(
            "Fixed screens require one universe available in the criterion table"
        )
    original = aggregate_patient_moments(
        table,
        min_independent_patients=min_independent_patients,
        min_weighted_support=min_weighted_support,
    )
    labels = tuple("utility:" + selection.policy for selection in selections) + (
        "contrast:error_transport-source_ccc",
        "contrast:synthetic_ccc-error_transport",
        "contrast:synthetic_ccc-source_ccc",
    )

    def statistic(counts: np.ndarray) -> dict[str, float]:
        profile = aggregate_patient_moments(
            table,
            counts,
            locked_support=original.support,
            min_weighted_support=min_weighted_support,
        )
        criterion = profile.ccc_mapping()
        fixed_criterion = {descriptor: criterion[descriptor] for descriptor in universe}
        evaluations = [
            evaluate_selection(selection, fixed_criterion) for selection in selections
        ]
        values = {"utility:" + item.policy: item.utility for item in evaluations}
        for contrast in utility_contrasts(evaluations):
            values[
                "contrast:" + str(contrast["first"]) + "-" + str(contrast["second"])
            ] = float(contrast["difference"])
        return values

    return conditional_bootstrap(
        patient_count=len(table.patient_ids),
        statistic_ids=labels,
        statistic=statistic,
        rng=rng,
        conditioning=AUDIT_CONDITIONING,
        replicates=replicates,
    )


def summarize_bootstrap(
    distribution: BootstrapDistribution,
    *,
    nominal_interval_allowed: bool = False,
    nominal_level: float = 0.95,
    max_undefined_fraction: float = 0.05,
    point_estimates: Mapping[str, float] | None = None,
) -> tuple[dict[str, object], ...]:
    """Report incomplete distributions without issuing an intact nominal interval.

    The caller enables nominal audit intervals only after applying the separately
    recorded coverage decision. Full-cohort ranges always remain stability summaries.
    """
    if not 0 < nominal_level < 1 or not 0 <= max_undefined_fraction <= 1:
        raise ValueError("Invalid predeclared interval rule")
    if point_estimates is not None and set(point_estimates) != set(
        distribution.statistic_ids
    ):
        raise ValueError(
            "Original point estimates must cover exactly the bootstrap statistic identities"
        )
    nominal = (
        nominal_interval_allowed and distribution.conditioning == AUDIT_CONDITIONING
    )
    rows = []
    for column, identifier in enumerate(distribution.statistic_ids):
        values = distribution.values[:, column]
        finite = values[np.isfinite(values)]
        failure_fraction = 1 - len(finite) / len(values)
        lower, upper = (
            np.quantile(finite, [(1 - nominal_level) / 2, (1 + nominal_level) / 2])
            if len(finite)
            else (nan, nan)
        )
        original_defined = point_estimates is None or np.isfinite(
            point_estimates[identifier]
        )
        permitted = (
            original_defined and failure_fraction <= max_undefined_fraction + 1e-15
        )
        kind = (
            "pointwise_nominal_percentile_interval"
            if nominal
            else "bootstrap_sampling_stability_range"
        )
        rows.append(
            {
                "statistic_id": identifier,
                "conditioning": distribution.conditioning,
                "replicates": len(values),
                "defined_replicates": len(finite),
                "undefined_fraction": failure_fraction,
                "nominal_level": nominal_level,
                "original_statistic_defined": None
                if point_estimates is None
                else bool(original_defined),
                "interval_kind": kind if permitted else "not_issued",
                "interval_lower": float(lower) if permitted else nan,
                "interval_upper": float(upper) if permitted else nan,
                "available_distribution_lower": float(lower),
                "available_distribution_upper": float(upper),
                "reason": "original_statistic_undefined"
                if not original_defined
                else None
                if permitted
                else "undefined_fraction_exceeds_declared_limit",
            }
        )
    return tuple(rows)
