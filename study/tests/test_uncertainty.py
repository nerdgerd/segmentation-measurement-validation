"""Bootstrap tests pin conditioning, clustered slots, and undefined reporting."""

import numpy as np
import pytest

from redesign2.empirical_math import prepare_patient_moments
from redesign2.policies import construct_screens
from redesign2.provenance import rng
from redesign2.uncertainty import (
    AUDIT_CONDITIONING,
    FULL_COHORT_CONDITIONING,
    BootstrapDistribution,
    conditional_bootstrap,
    fixed_selection_bootstrap,
    patient_bootstrap_multiplicities,
    summarize_bootstrap,
)


def test_rng(*key):
    return rng(
        {
            "randomness": {
                "master_seed": 20260915,
                "namespace": "redesign2_v1",
                "streams": {"smoke": "redesign2_v1_smoke"},
            }
        },
        "smoke",
        "numerical_test",
        *key,
    )


test_rng.__test__ = False


def test_patient_slot_resampling_is_reproducible_and_preserves_total_sample_size():
    first = patient_bootstrap_multiplicities(11, 100, rng=test_rng("slots"))
    second = patient_bootstrap_multiplicities(11, 100, rng=test_rng("slots"))
    assert np.array_equal(first, second)
    assert np.all(first.sum(axis=1) == 11)
    assert np.any((first > 0).sum(axis=1) < 10)


def test_fixed_selections_bootstrap_uses_shared_patients_and_never_reranks():
    x = np.repeat(np.arange(11, dtype=float)[:, None], 8, axis=1)
    w = x + np.linspace(0.1, 1.5, 8)
    table = prepare_patient_moments(
        x,
        w,
        [f"p{i:02}" for i in range(11)],
        ["u"] * 11,
        ["r"] * 11,
        eligible=np.ones_like(x, dtype=bool),
        descriptor_ids=list("abcdefgh"),
    )
    scores = dict(zip("abcdefg", range(7)))
    selections = construct_screens(
        source_ccc=scores, synthetic_ccc=scores, error_transport=scores
    )
    selected_before = tuple(item.selected_ids for item in selections)
    distribution = fixed_selection_bootstrap(
        table, selections, rng=test_rng("fixed"), replicates=100
    )
    assert distribution.conditioning == AUDIT_CONDITIONING
    assert tuple(item.selected_ids for item in selections) == selected_before
    assert np.all(distribution.values[:, 3:] == 0)
    assert np.all(np.isfinite(distribution.values[:, :3]))
    assert np.std(distribution.values[:, 0]) > 0


def test_full_cohort_callback_receives_one_shared_target_resample_for_all_statistics():
    calls = []

    def statistic(counts):
        calls.append(counts.copy())
        value = float(counts @ np.arange(12))
        return {"one": value, "two": value + 1}

    distribution = conditional_bootstrap(
        patient_count=12,
        statistic_ids=["one", "two"],
        statistic=statistic,
        rng=test_rng("callback"),
        conditioning=FULL_COHORT_CONDITIONING,
        replicates=20,
    )
    assert len(calls) == 20
    assert np.all(distribution.values[:, 1] - distribution.values[:, 0] == 1)
    summary = summarize_bootstrap(distribution, nominal_interval_allowed=True)
    assert all(
        row["interval_kind"] == "bootstrap_sampling_stability_range" for row in summary
    )


def test_undefined_fraction_withholds_interval_and_keeps_available_distribution():
    values = np.array([[0], [1], [np.nan], [np.nan]], dtype=float)
    distribution = BootstrapDistribution(
        ("utility",), values, AUDIT_CONDITIONING, 11, 4
    )
    summary = summarize_bootstrap(distribution, nominal_interval_allowed=True)[0]
    assert summary["undefined_fraction"] == 0.5
    assert np.isnan(summary["interval_lower"])
    assert np.isfinite(summary["available_distribution_lower"])
    assert summary["reason"] == "undefined_fraction_exceeds_declared_limit"
    assert summary["interval_kind"] == "not_issued"
    all_invalid = BootstrapDistribution(
        ("utility",), np.full((4, 1), np.nan), AUDIT_CONDITIONING, 11, 4
    )
    assert np.isnan(summarize_bootstrap(all_invalid)[0]["available_distribution_lower"])


def test_original_undefined_statistic_cannot_receive_reselection_interval():
    distribution = BootstrapDistribution(
        ("utility",), np.array([[0.1], [0.2], [0.3]]), FULL_COHORT_CONDITIONING, 11, 3
    )
    summary = summarize_bootstrap(distribution, point_estimates={"utility": np.nan})[0]
    assert summary["undefined_fraction"] == 0
    assert summary["original_statistic_defined"] is False
    assert summary["interval_kind"] == "not_issued"
    assert summary["reason"] == "original_statistic_undefined"
    assert np.isnan(summary["interval_lower"])
    assert np.isfinite(summary["available_distribution_lower"])


def test_callback_cannot_silently_change_statistic_set():
    with pytest.raises(ValueError, match="changed"):
        conditional_bootstrap(
            patient_count=11,
            statistic_ids=["utility"],
            statistic=lambda counts: {"replacement": 0.0},
            rng=test_rng("identity"),
            conditioning=FULL_COHORT_CONDITIONING,
            replicates=1,
        )
