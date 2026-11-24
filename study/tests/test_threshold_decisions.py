import hashlib
import json
from math import isnan

import numpy as np
import pandas as pd

from redesign2.threshold_decisions import (
    ALL_METRICS,
    decision_contrasts,
    retained_decisions,
    summarize_replicates,
    threshold_decision,
)
from redesign2.uncertainty import patient_bootstrap_multiplicities


def test_equality_is_retained_and_counts_partition_fixed_battery():
    result = threshold_decision(
        {"a": 0.85, "b": 0.84}, {"a": 0.85, "b": 0.90}, threshold=0.85
    )
    assert result.labels == {"a": "TP", "b": "FP"}
    assert result.values["n_target_retained"] == 1
    assert sum(result.values[name] for name in ("tp", "fp", "fn", "tn")) == 2


def test_empty_candidate_and_reference_denominators_remain_undefined():
    empty_candidate = threshold_decision({"a": 0.9}, {"a": 0.2}, threshold=0.8)
    assert isnan(empty_candidate.values["false_retention_proportion"])
    assert (
        empty_candidate.reasons["false_retention_proportion"]
        == "empty_candidate_retained_set"
    )
    empty_reference = threshold_decision({"a": 0.2}, {"a": 0.9}, threshold=0.8)
    assert isnan(empty_reference.values["false_exclusion_proportion"])
    both_empty = threshold_decision({"a": 0.2}, {"a": 0.3}, threshold=0.8)
    assert isnan(both_empty.values["jaccard"])
    assert both_empty.reasons["jaccard"] == "both_retained_sets_empty"


def test_negative_ccc_is_a_valid_rejection_not_missing():
    result = threshold_decision(
        {"a": -0.5, "b": 0.9}, {"a": -0.2, "b": 0.8}, threshold=0.75
    )
    assert result.labels["a"] == "TN"
    assert all(
        isnan(value) is False
        for metric, value in result.values.items()
        if metric != "false_retention_proportion" or result.values["n_method_retained"]
    )


def test_undefined_ccc_does_not_become_rejection_or_shrink_support():
    result = threshold_decision(
        {"a": np.nan, "b": 0.9}, {"a": 0.1, "b": 0.9}, threshold=0.8
    )
    assert all(isnan(result.values[metric]) for metric in ALL_METRICS)
    assert result.labels == {"a": "UNDEFINED", "b": "UNDEFINED"}
    assert "undefined_observed_target_ccc:1" in result.reasons["tp"]


def test_descriptor_order_does_not_change_metrics_or_labels():
    observed = {"a": 0.7, "b": 0.8, "c": 0.9}
    candidate = {"a": 0.9, "b": 0.7, "c": 0.9}
    first = threshold_decision(
        observed, candidate, threshold=0.8, descriptor_ids=("a", "b", "c")
    )
    second = threshold_decision(
        observed, candidate, threshold=0.8, descriptor_ids=("c", "a", "b")
    )
    assert first.values == second.values
    assert first.labels == second.labels


def test_contrast_sign_is_synthetic_minus_source():
    observed = {"a": 0.9, "b": 0.9, "c": 0.1}
    synthetic = threshold_decision(
        observed, {"a": 0.9, "b": 0.9, "c": 0.1}, threshold=0.8
    )
    source = threshold_decision(observed, {"a": 0.9, "b": 0.1, "c": 0.9}, threshold=0.8)
    contrast, reasons = decision_contrasts(synthetic, source)
    assert contrast["jaccard"] > 0
    assert contrast["decision_disagreement_rate"] < 0
    assert all(reason is None for reason in reasons.values())


def test_audit_candidate_decisions_are_independent_of_audit_alternates():
    candidate = {"a": 0.91, "b": 0.79}
    locked = retained_decisions(candidate, threshold=0.8, descriptor_ids=("a", "b"))
    for audit_outcomes in (
        {"a": 0.1, "b": 0.9},
        {"a": 0.9, "b": 0.1},
        {"a": np.nan, "b": np.nan},
    ):
        assert (
            retained_decisions(
                candidate, threshold=0.8, descriptor_ids=tuple(audit_outcomes)
            )
            == locked
        )


def test_patient_bootstrap_samples_cluster_multiplicities_reproducibly():
    first = patient_bootstrap_multiplicities(4, 20, rng=np.random.default_rng(12))
    second = patient_bootstrap_multiplicities(4, 20, rng=np.random.default_rng(12))
    assert np.array_equal(first, second)
    assert np.all(first.sum(axis=1) == 4)
    # Downstream moments multiply whole patient rows; no unit/alternate index is sampled here.
    assert first.shape == (20, 4)


def test_fixed_seed_reproduces_summaries_and_serialized_hash():
    def build(seed):
        generator = np.random.default_rng(seed)
        values = generator.normal(size=100)
        summary = summarize_replicates(values, [None] * len(values), point=0.0)
        frame = pd.DataFrame([summary])
        frame["bootstrap_undefined_reasons"] = frame.bootstrap_undefined_reasons.map(
            lambda value: json.dumps(value, sort_keys=True)
        )
        return summary, hashlib.sha256(
            frame.to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest()

    first, first_hash = build(7)
    second, second_hash = build(7)
    assert first == second
    assert first_hash == second_hash


def test_more_than_five_percent_undefined_withholds_range():
    values = [1.0] * 94 + [np.nan] * 6
    reasons = [None] * 94 + ["empty_candidate_retained_set"] * 6
    summary = summarize_replicates(values, reasons, point=1.0)
    assert summary["interval_status"] == "withheld_undefined_gt_5pct"
    assert isnan(summary["bootstrap_p025"])
    assert summary["bootstrap_undefined_reasons"] == {"empty_candidate_retained_set": 6}


def test_retained_counts_are_monotone_across_locked_grid():
    scores = {"a": 0.76, "b": 0.82, "c": 0.93, "d": 0.96}
    counts = [
        sum(
            retained_decisions(
                scores, threshold=value, descriptor_ids=tuple(scores)
            ).values()
        )
        for value in (0.75, 0.80, 0.85, 0.90, 0.95)
    ]
    assert all(first >= second for first, second in zip(counts, counts[1:]))


def test_support_mismatch_and_repeated_identity_are_rejected():
    with np.testing.assert_raises(ValueError):
        threshold_decision({"a": 0.8}, {"b": 0.8}, threshold=0.8)
    with np.testing.assert_raises(ValueError):
        threshold_decision(
            {"a": 0.8}, {"a": 0.8}, threshold=0.8, descriptor_ids=("a", "a")
        )
