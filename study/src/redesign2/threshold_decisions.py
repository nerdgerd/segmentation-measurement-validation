"""Threshold decisions compare fixed descriptor identities without rank substitution.

An undefined CCC makes the affected decision cell undefined. It is never
converted to a rejection, removed from the battery, or replaced after the
cutoff is applied. This keeps the locked common support visible in every count.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, nan
from typing import Mapping, Sequence

import numpy as np


COUNT_METRICS = (
    "n_target_retained",
    "n_method_retained",
    "tp",
    "fp",
    "fn",
    "tn",
    "net_retained_count_difference",
)
RATE_METRICS = (
    "false_retention_proportion",
    "false_exclusion_proportion",
    "jaccard",
    "decision_disagreement_rate",
)
ALL_METRICS = COUNT_METRICS + RATE_METRICS
CONTRAST_METRICS = (
    "jaccard",
    "false_retention_proportion",
    "false_exclusion_proportion",
    "decision_disagreement_rate",
)


@dataclass(frozen=True)
class ThresholdDecision:
    values: Mapping[str, float]
    reasons: Mapping[str, str | None]
    labels: Mapping[str, str]


def retained_decisions(
    scores: Mapping[str, float], *, threshold: float, descriptor_ids: Sequence[str]
) -> dict[str, bool] | None:
    """Apply a cutoff to one locked score map without access to comparator outcomes."""
    ids = tuple(descriptor_ids)
    if not ids or len(set(ids)) != len(ids) or set(scores) != set(ids):
        raise ValueError(
            "Scores must cover one fixed battery of unique descriptor identities"
        )
    if not isfinite(threshold):
        raise ValueError("The CCC cutoff must be finite")
    if any(not isfinite(float(scores[item])) for item in ids):
        return None
    return {item: float(scores[item]) >= threshold for item in ids}


def _identities(
    observed: Mapping[str, float],
    candidate: Mapping[str, float],
    descriptor_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    ids = (
        tuple(descriptor_ids) if descriptor_ids is not None else tuple(sorted(observed))
    )
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(
            "A nonempty fixed battery of unique descriptor identities is required"
        )
    if set(observed) != set(ids) or set(candidate) != set(ids):
        raise ValueError(
            "Observed and candidate scores must share the fixed eligible support"
        )
    return ids


def threshold_decision(
    observed: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    threshold: float,
    descriptor_ids: Sequence[str] | None = None,
) -> ThresholdDecision:
    """Classify one candidate against the observed target reference decision."""
    if not isfinite(threshold):
        raise ValueError("The CCC cutoff must be finite")
    ids = _identities(observed, candidate, descriptor_ids)
    missing_observed = [item for item in ids if not isfinite(float(observed[item]))]
    missing_candidate = [item for item in ids if not isfinite(float(candidate[item]))]
    if missing_observed or missing_candidate:
        parts = []
        if missing_observed:
            parts.append(f"undefined_observed_target_ccc:{len(missing_observed)}")
        if missing_candidate:
            parts.append(f"undefined_candidate_ccc:{len(missing_candidate)}")
        reason = ";".join(parts)
        return ThresholdDecision(
            {metric: nan for metric in ALL_METRICS},
            {metric: reason for metric in ALL_METRICS},
            {item: "UNDEFINED" for item in ids},
        )

    target_decisions = retained_decisions(
        observed, threshold=threshold, descriptor_ids=ids
    )
    method_decisions = retained_decisions(
        candidate, threshold=threshold, descriptor_ids=ids
    )
    target = {item for item, retained in target_decisions.items() if retained}
    method = {item for item, retained in method_decisions.items() if retained}
    tp, fp, fn = len(method & target), len(method - target), len(target - method)
    tn = len(ids) - tp - fp - fn
    union = len(method | target)
    values = {
        "n_target_retained": float(len(target)),
        "n_method_retained": float(len(method)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "net_retained_count_difference": float(len(method) - len(target)),
        "false_retention_proportion": fp / len(method) if method else nan,
        "false_exclusion_proportion": fn / len(target) if target else nan,
        "jaccard": tp / union if union else nan,
        "decision_disagreement_rate": (fp + fn) / len(ids),
    }
    reasons = {metric: None for metric in ALL_METRICS}
    if not method:
        reasons["false_retention_proportion"] = "empty_candidate_retained_set"
    if not target:
        reasons["false_exclusion_proportion"] = "empty_observed_target_retained_set"
    if not union:
        reasons["jaccard"] = "both_retained_sets_empty"
    labels = {}
    for item in ids:
        labels[item] = (
            "TP"
            if item in method and item in target
            else "FP"
            if item in method
            else "FN"
            if item in target
            else "TN"
        )
    return ThresholdDecision(values, reasons, labels)


def decision_contrasts(
    synthetic: ThresholdDecision, source: ThresholdDecision
) -> tuple[dict[str, float], dict[str, str | None]]:
    """Return synthetic-minus-source contrasts with the prespecified sign."""
    values, reasons = {}, {}
    for metric in CONTRAST_METRICS:
        first, second = synthetic.values[metric], source.values[metric]
        if isfinite(float(first)) and isfinite(float(second)):
            values[metric] = float(first) - float(second)
            reasons[metric] = None
        else:
            values[metric] = nan
            parts = [
                value
                for value in (synthetic.reasons[metric], source.reasons[metric])
                if value
            ]
            reasons[metric] = (
                ";".join(dict.fromkeys(parts)) or "undefined_component_metric"
            )
    return values, reasons


def summarize_replicates(
    values: Sequence[float],
    reasons: Sequence[str | None],
    *,
    point: float,
    max_undefined_fraction: float = 0.05,
    nominal_level: float = 0.95,
) -> dict[str, object]:
    """Summarize a conditional bootstrap without upgrading it to a confidence interval."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != len(reasons) or not len(array):
        raise ValueError("Aligned nonempty replicate values and reasons are required")
    finite = array[np.isfinite(array)]
    undefined = len(array) - len(finite)
    fraction = undefined / len(array)
    counts = Counter(
        reason or "unspecified_undefined_reason"
        for value, reason in zip(array, reasons)
        if not isfinite(float(value))
    )
    if not isfinite(float(point)):
        status = "withheld_original_undefined"
    elif fraction > max_undefined_fraction + 1e-15:
        status = "withheld_undefined_gt_5pct"
    elif not len(finite):
        status = "withheld_no_defined_replicates"
    else:
        status = "reported"
    alpha = (1 - nominal_level) / 2
    lower, upper = (
        np.quantile(finite, [alpha, 1 - alpha]) if len(finite) else (nan, nan)
    )
    report = status == "reported"
    return {
        "point_estimate": float(point),
        "bootstrap_p025": float(lower) if report else nan,
        "bootstrap_p975": float(upper) if report else nan,
        "bootstrap_available_p025": float(lower),
        "bootstrap_available_p975": float(upper),
        "bootstrap_replicates_requested": int(len(array)),
        "bootstrap_replicates_defined": int(len(finite)),
        "bootstrap_replicates_undefined": int(undefined),
        "bootstrap_undefined_fraction": float(fraction),
        "bootstrap_undefined_reasons": dict(sorted(counts.items())),
        "interval_status": status,
        "interval_kind": "conditional_bootstrap_sampling_stability_range"
        if report
        else "not_issued",
    }
