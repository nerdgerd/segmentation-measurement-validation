from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from redesign2.simulation import (
    PopulationLaw,
    Template,
    coefficient_battery,
    make_laws,
    population_moments,
    population_transport,
    coverage_fixture,
    hierarchical_draws,
    demonstration_template,
    noisy_reference_moments,
)
from redesign2.simulation_driver import population_tables, smoke_tables


@pytest.fixture
def cfg():
    return yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "protocol.yaml").read_text()
    )


@pytest.fixture
def battery(cfg):
    template = demonstration_template()
    return coefficient_battery(
        cfg,
        template.descriptor_ids,
        template.families,
        template_key=template.name,
        stream="smoke",
    )


@pytest.mark.parametrize("regime", ["additive", "biased_proportional"])
@pytest.mark.parametrize(
    "multiplier,shift", [(0.5, 0), (0.5, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
)
def test_exact_population_transport_under_preserved_law(
    battery, regime, multiplier, shift
):
    real, _ = make_laws(battery, h=1.0, theta=0.5, regime=regime)
    mu = shift * np.sqrt(real.source_variance)
    v = multiplier * real.source_variance
    source = population_moments(real)
    truth = population_moments(real, mu, v)
    predictions = population_transport(source, mu, v)
    for field in [
        "ccc",
        "disagreement_mean",
        "disagreement_variance",
        "reference_disagreement_covariance",
        "disagreement_mse",
    ]:
        np.testing.assert_allclose(
            [getattr(p, field) for p in predictions],
            truth[field],
            atol=1e-13,
            rtol=1e-13,
        )


def test_heteroscedastic_truth_matches_independent_gaussian_quadrature_and_exposes_failure(
    battery,
):
    real, _ = make_laws(battery, h=1.0, theta=0.0, regime="heteroscedastic")
    mu, v = np.sqrt(real.source_variance), 2 * real.source_variance
    truth = population_moments(real, mu, v)
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    x = mu + np.sqrt(2 * v) * nodes[:, None]
    conditional_variance = real.residual_sd**2 * np.exp(
        0.6 * (x - real.source_mean) / np.sqrt(real.source_variance)
    )
    integrated = np.sum(weights[:, None] * conditional_variance, axis=0) / np.sqrt(
        np.pi
    )
    np.testing.assert_allclose(truth["residual_second_moment"], integrated, rtol=1e-13)
    predictions = population_transport(population_moments(real), mu, v)
    assert (
        np.max(np.abs(np.asarray([p.ccc for p in predictions]) - truth["ccc"])) > 0.001
    )
    # This law deliberately keeps source centering; target recentering is a different law.
    incorrectly_recentered = replace(real, source_mean=mu, source_variance=v)
    assert not np.allclose(
        population_moments(incorrectly_recentered)["residual_second_moment"], integrated
    )


def test_coefficients_are_identity_keyed_and_case_mix_holds_disagreement_fixed(
    cfg, battery
):
    reverse = coefficient_battery(
        cfg,
        battery.descriptor_id.to_list()[::-1],
        battery.family.to_list()[::-1],
        template_key="fixed_demonstration_v1",
        stream="smoke",
    )
    np.testing.assert_array_equal(
        battery.iloc[:, 2:].to_numpy(), reverse.iloc[::-1, 2:].to_numpy()
    )
    first, _ = make_laws(battery, h=0, theta=0.5, regime="biased_proportional")
    second, _ = make_laws(battery, h=1.5, theta=0.5, regime="biased_proportional")
    for key in ["bias", "slope", "residual_sd"]:
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
    assert not np.array_equal(first.source_variance, second.source_variance)
    real, syn = make_laws(battery, h=1, theta=1, regime="additive")
    np.testing.assert_array_equal(real.residual_sd, syn.residual_sd)


@pytest.mark.parametrize("regime", ["additive", "heteroscedastic"])
def test_coverage_fixture_has_exact_equal_and_signed_population_utilities(
    cfg, battery, regime
):
    real, _ = make_laws(battery, h=1, theta=0, regime=regime)
    fixture, sets, utilities = coverage_fixture(real, battery.descriptor_id.to_list())
    assert utilities["second"] == utilities["first"]
    assert utilities["higher"] > utilities["first"] > utilities["lower"]
    assert set(sets["first"]).isdisjoint(sets["second"])
    assert all(len(selected) == 3 for selected in sets.values())
    data = hierarchical_draws(
        cfg,
        demonstration_template(),
        fixture,
        fixture,
        stream="smoke",
        replicate=0,
        draw_count=3,
    )
    # Exact equal population truth does not make the observed estimators identical.
    assert not np.array_equal(data["real"][:, :3], data["real"][:, 3:6])


def test_hierarchical_draws_preserve_reference_and_patient_weights(cfg):
    template = Template(
        "unequal_structure",
        ("a", "b"),
        ("f", "f"),
        ("p1", "p1", "p2"),
        ("n1", "n2", "n1"),
        (1, 3, 2),
    )
    law = PopulationLaw(np.zeros(2), np.ones(2), np.zeros(2), np.zeros(2), np.ones(2))
    first = hierarchical_draws(
        cfg, template, law, law, stream="smoke", replicate=0, draw_count=5
    )
    second = hierarchical_draws(
        cfg, template, law, law, stream="smoke", replicate=0, draw_count=5
    )
    np.testing.assert_array_equal(first["real"], second["real"])
    np.testing.assert_allclose(first["unit_weights"], [0.25, 0.25, 0.5])
    np.testing.assert_allclose(
        first["real_weights"], [0.25, 1 / 12, 1 / 12, 1 / 12, 0.25, 0.25]
    )
    assert first["synthetic"].shape == (3, 5, 2)
    assert first["real"].shape == (6, 2)
    assert len(first["reference"]) == 3
    np.testing.assert_array_equal(first["real_unit_index"], [0, 1, 1, 1, 2, 2])


def test_template_refuses_inferred_families_and_duplicate_units():
    with pytest.raises(ValueError, match="family"):
        Template("x", ("a",), ("",), ("p",), ("n",), (1,)).validate()
    with pytest.raises(ValueError, match="more than once"):
        Template("x", ("a",), ("f",), ("p", "p"), ("n", "n"), (1, 1)).validate()


def test_noisy_reference_keeps_observed_disagreement_and_latent_error_distinct():
    data = noisy_reference_moments(np.array([0.1, 1.0, 10.0]))
    np.testing.assert_allclose(data.real_observed_disagreement_mse, 0.34)
    np.testing.assert_allclose(data.real_latent_error_mse, 0.25)
    np.testing.assert_allclose(data.synthetic_observed_disagreement_mse, 0.25)
    np.testing.assert_allclose(data.synthetic_latent_error_mse, 0.34)
    np.testing.assert_allclose(data.real_reference_disagreement_covariance, -0.09)


def test_population_driver_keeps_success_and_failure_cells_and_both_contrasts(cfg):
    template = demonstration_template()
    tables = population_tables(
        cfg,
        template.descriptor_ids,
        template.families,
        template_key=template.name,
        template_role="artificial_dimensionless_demonstration",
        stream="smoke",
    )
    assert len(tables["population_mechanism_summary"]) == 27
    stress = tables["transport_stress_tests"]
    invariant = stress[
        (stress.regime != "heteroscedastic")
        & (stress.scenario == "unchanged_conditional_law")
    ]
    assert invariant.error_transport_absolute_ccc_error.max() < 1e-12
    assert "changed_conditional_law" in set(stress.scenario)
    assert "heteroscedastic" in set(stress.regime)
    assert {
        "error_transport_minus_source_ccc",
        "synthetic_ccc_minus_error_transport",
    } <= set(tables["population_policy_utility"])


def test_smoke_driver_labels_artificial_nonfinal_outputs(cfg):
    data = smoke_tables(cfg, replicates=1)["numerical_smoke"]
    assert len(data) == 36
    assert not data.final_simulation_estimate.any()
    assert set(data.template_role) == {"artificial_numerical_smoke"}
