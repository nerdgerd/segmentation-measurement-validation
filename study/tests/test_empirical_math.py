"""Cached patient moments must equal explicit clustered-table integration."""

import numpy as np
import pytest

from redesign2.empirical_math import (
    aggregate_patient_moments,
    batch_joint_moments,
    prepare_patient_moments,
)
from redesign2.moments import hierarchical_weights, weighted_joint_moments


def test_cached_moments_equal_expanded_unequal_nodule_reader_support():
    patients = ["a", "a", "b", "b", "b", "c"]
    units = ["u", "u", "u", "v", "v", "u"]
    readers = ["r1", "r2", "r1", "r1", "r2", "r1"]
    x = np.array([[1, 2], [1, 2], [3, 1], [2, 3], [2, 3], [4, 0]], dtype=float)
    w = x + np.array(
        [[0.1, 0.2], [-0.2, 0.3], [0.4, -0.3], [0.2, 0.3], [-0.1, -0.4], [0.5, 0.1]]
    )
    mask = np.ones_like(x, dtype=bool)
    mask[1, 0] = False
    profile = batch_joint_moments(
        x,
        w,
        patients,
        units,
        readers,
        eligible=mask,
        descriptor_ids=["shape", "size"],
        min_independent_patients=1,
        min_weighted_support=0.8,
    )
    for column, moments in enumerate(profile.moments):
        support = hierarchical_weights(
            patients, units, readers, eligible=mask[:, column]
        )
        kept = mask[:, column]
        direct = weighted_joint_moments(
            x[kept, column], w[kept, column], np.asarray(support.weights)[kept]
        )
        for field in (
            "reference_mean",
            "reference_variance",
            "disagreement_mean",
            "disagreement_variance",
            "reference_disagreement_covariance",
            "disagreement_mse",
            "ccc",
        ):
            assert getattr(moments, field) == pytest.approx(
                getattr(direct, field), abs=2e-15
            )
        assert profile.support[column].original_weighted_support == pytest.approx(
            support.original_support_coverage
        )


def test_original_patient_gate_is_not_reapplied_to_distinct_bootstrap_originals():
    x = np.arange(11, dtype=float)[:, None]
    w = x + 0.2
    table = prepare_patient_moments(
        x,
        w,
        [f"p{i:02}" for i in range(11)],
        ["u"] * 11,
        ["r"] * 11,
        eligible=np.ones_like(x, dtype=bool),
        descriptor_ids=["j"],
    )
    original = aggregate_patient_moments(table)
    repeated = aggregate_patient_moments(
        table, [5, 6] + [0] * 9, locked_support=original.support
    )
    assert repeated.support[0].independent_patients == 11
    assert repeated.support[0].evaluable
    assert np.isfinite(repeated.moments[0].ccc)
    direct = weighted_joint_moments(
        [0] * 5 + [1] * 6, [0.2] * 5 + [1.2] * 6, [1 / 11] * 11
    )
    assert repeated.moments[0].ccc == pytest.approx(direct.ccc)


def test_missing_support_remains_a_replicate_failure_without_changing_universe():
    x = np.arange(11, dtype=float)[:, None]
    mask = np.ones_like(x, dtype=bool)
    mask[-1] = False
    table = prepare_patient_moments(
        x,
        x + 0.2,
        [f"p{i:02}" for i in range(11)],
        ["u"] * 11,
        ["r"] * 11,
        eligible=mask,
        descriptor_ids=["j"],
    )
    original = aggregate_patient_moments(table)
    assert original.support[0].evaluable
    failed = aggregate_patient_moments(
        table, [2, 3] + [0] * 8 + [6], locked_support=original.support
    )
    assert failed.descriptor_ids == original.descriptor_ids
    assert failed.support[0].reason == "insufficient_original_weighted_support"
    assert np.isnan(failed.moments[0].ccc)
    assert np.isfinite(failed.moments[0].disagreement_mse)


def test_ineligible_nonfinite_values_are_explicit_and_original_failure_is_retained():
    x = np.arange(10, dtype=float)[:, None]
    w = x + 0.2
    w[-1] = np.nan
    mask = np.ones_like(x, dtype=bool)
    with pytest.raises(ValueError, match="eligible measurement"):
        prepare_patient_moments(
            x,
            w,
            [f"p{i}" for i in range(10)],
            ["u"] * 10,
            ["r"] * 10,
            eligible=mask,
            descriptor_ids=["j"],
        )
    mask[-1] = False
    table = prepare_patient_moments(
        x,
        w,
        [f"p{i}" for i in range(10)],
        ["u"] * 10,
        ["r"] * 10,
        eligible=mask,
        descriptor_ids=["j"],
    )
    original = aggregate_patient_moments(table)
    assert original.support[0].reason == "insufficient_independent_patients"
    repeated = aggregate_patient_moments(
        table, [5, 5] + [0] * 8, locked_support=original.support
    )
    assert repeated.support[0].reason == "insufficient_independent_patients"
    assert np.isnan(repeated.moments[0].ccc)
