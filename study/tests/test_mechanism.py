"""Check law coherence, null clustering and calibration without favorable assertions."""

import numpy as np
import pandas as pd
from dataclasses import replace
from redesign2.mechanism import (
    match_law,
    permutation_indices,
    permuted_associations,
    coverage_prediction,
    top_k,
)
from redesign2.simulation import coefficient_battery, population_moments, make_laws

CFG = {
    "randomness": {
        "master_seed": 2026091702,
        "namespace": "test_critical_revision",
        "streams": {"mechanism_final": "coefficients"},
    }
}


def battery():
    return coefficient_battery(
        CFG, [f"d{i}" for i in range(24)], ["f"] * 24, template_key="test"
    )


def test_matching_preserves_all_disagreement_components_and_marginals():
    b = battery()
    law, _ = make_laws(b, h=0.7, theta=0, regime="biased_proportional")
    law = replace(law, source_mean=np.ones(24))
    m = population_moments(law)
    got, r = match_law(
        b,
        np.ones(24),
        law.source_variance,
        m["disagreement_mse"],
        law="biased_proportional",
        theta=0,
        side="observed",
    )
    assert r["matched"]
    assert abs(r["attained_reference_log_iqr"] - r["reference_log_iqr"]) < 1e-6
    assert abs(r["attained_disagreement_log_iqr"] - r["disagreement_log_iqr"]) < 1e-6
    multiplier = got.residual_sd / law.residual_sd
    np.testing.assert_allclose(got.bias, law.bias * multiplier)
    np.testing.assert_allclose(got.slope, law.slope * multiplier)


def test_matching_is_descriptorwise_unit_invariant():
    b = battery()
    law, _ = make_laws(b, h=0.8, theta=0, regime="heteroscedastic")
    m = population_moments(law)
    units = np.exp(np.linspace(-8, 8, 24))
    a, r = match_law(
        b,
        np.ones(24),
        law.source_variance,
        m["disagreement_mse"],
        law="heteroscedastic",
        theta=0,
        side="observed",
    )
    c, s = match_law(
        b,
        units,
        law.source_variance * units**2,
        m["disagreement_mse"] * units**2,
        law="heteroscedastic",
        theta=0,
        side="observed",
    )
    assert r["matched"] == s["matched"]
    np.testing.assert_allclose(
        population_moments(a)["ccc"], population_moments(c)["ccc"], rtol=1e-10
    )


def test_permutation_carries_repeated_regions_and_blocks():
    ids = ["WT__a", "TC__a", "WT__b", "TC__b", "WT__c", "TC__c", "vol_ET_cm3"]
    idx, blocks = permutation_indices(
        ids,
        ["f"] * 6 + ["volume"],
        ["WT", "TC"] * 3 + ["ET"],
        repetitions=100,
        rng=np.random.default_rng(5),
    )
    np.testing.assert_array_equal(idx[:, ::2][:, :3] // 2, idx[:, 1:6:2] // 2)
    assert (idx[:, -1] == 6).all()
    assert sorted(int(r["distinct_permutations"]) for r in blocks) == [1, 6]
    assert all(set(row) == set(range(7)) for row in idx)


def test_null_identity_reconstructs_CCC_without_breaking_tuple():
    b = battery()
    law, _ = make_laws(b, h=1, theta=0, regime="heteroscedastic")
    m = population_moments(law)
    idx = np.arange(24)[None, :]
    rho = permuted_associations(
        m["ccc"],
        m["reference_variance"],
        m["reference_disagreement_covariance"],
        m["disagreement_mse"],
        idx,
    )
    np.testing.assert_allclose(rho, [1])


def test_normal_location_approximation_and_undefined_selection():
    c = coverage_prediction(np.array([0, 0.5, 1, np.nan]))
    np.testing.assert_allclose(c[0], 0.95, atol=1e-7)
    assert c[0] > c[1] > c[2] and np.isnan(c[3])
    assert top_k(("b", "a", "c", "d"), np.array([1, 1, 0, 0])) == (1,)
    assert top_k(("a", "b"), np.array([1, np.nan])) is None


def test_normal_centres_match_location_model_without_interval_replacement():
    z = np.random.default_rng(912).standard_normal(250000)
    for bias in [0, 0.3, 0.7]:
        empirical = np.mean(abs(z + 2 * bias) <= 1.959964)
        assert abs(empirical - float(coverage_prediction(bias))) < 0.003


def test_full_matched_law_moments_include_bias_slope_and_heteroscedasticity():
    b = battery()
    base, _ = make_laws(b, h=0.65, theta=0, regime="heteroscedastic")
    reference = np.exp(np.linspace(-2, 2, 24))
    means = np.ones(24)
    D = (
        base.bias**2
        + base.slope**2 * reference
        + base.residual_sd**2 * np.exp(2 * 0.3**2)
    )
    law, record = match_law(
        b, means, reference, D, law="heteroscedastic", theta=0, side="observed"
    )
    assert record["matched"]
    expected_a = law.slope * law.source_variance
    expected_D = (
        law.bias**2
        + law.slope**2 * law.source_variance
        + law.residual_sd**2 * np.exp(2 * 0.3**2)
    )
    expected_ccc = (
        2
        * (law.source_variance + expected_a)
        / (2 * (law.source_variance + expected_a) + expected_D)
    )
    actual = population_moments(law)
    np.testing.assert_allclose(actual["reference_disagreement_covariance"], expected_a)
    np.testing.assert_allclose(actual["disagreement_mse"], expected_D)
    np.testing.assert_allclose(actual["ccc"], expected_ccc)
