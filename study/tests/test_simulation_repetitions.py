from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from redesign2.moments import hierarchical_weights, weighted_joint_moments
from redesign2.simulation import (
    Template,
    coefficient_battery,
    make_laws,
    hierarchical_draws,
)
from redesign2.simulation_repetitions import (
    SimulationSupport,
    PatientMoments,
    finite_sample_tables,
    coverage_tables,
    summarize_coverage,
)


@pytest.fixture
def cfg():
    return yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "protocol.yaml").read_text()
    )


@pytest.fixture
def template():
    return Template(
        "artificial_11_patient_fixture",
        tuple(f"d{i}" for i in range(8)),
        tuple(f"f{i // 4}" for i in range(8)),
        tuple(f"p{i:02d}" for i in range(11)),
        tuple("n" for _ in range(11)),
        tuple(2 for _ in range(11)),
    )


@pytest.fixture
def support(template):
    return SimulationSupport(
        np.ones((11, 8), dtype=bool),
        np.ones((22, 8), dtype=bool),
        np.ones((11, 100, 8), dtype=bool),
    )


def test_patient_sufficient_moments_match_explicit_core_weights(cfg, template, support):
    battery = coefficient_battery(
        cfg,
        template.descriptor_ids,
        template.families,
        template_key=template.name,
        stream="smoke",
    )
    law, syn = make_laws(battery, h=1, theta=0, regime="heteroscedastic")
    data = hierarchical_draws(cfg, template, law, syn, stream="smoke", replicate=0)
    support.real_available[0, 0] = False
    unit_index = np.repeat(np.arange(11), 2)
    accumulator = PatientMoments(
        template, unit_index, support.real_available, support.reference_available
    )
    stats = accumulator.integrate(data["reference"], data["real"])
    actual = accumulator.pool(stats)
    patients = [template.patients[i] for i in unit_index]
    nodules = [template.nodules[i] for i in unit_index]
    weights = hierarchical_weights(
        patients,
        nodules,
        [str(i % 2) for i in range(22)],
        eligible=support.real_available[:, 0].tolist(),
    )
    expected = weighted_joint_moments(
        data["reference"][unit_index, 0], data["real"][:, 0], weights.weights
    )
    for field in [
        "reference_mean",
        "reference_variance",
        "disagreement_mean",
        "disagreement_variance",
        "reference_disagreement_covariance",
        "disagreement_mse",
        "ccc",
    ]:
        np.testing.assert_allclose(
            actual[field][0, 0], getattr(expected, field), atol=1e-12, rtol=1e-12
        )


def test_bootstrap_samples_slots_without_invalid_distinct_original_gate(
    cfg, template, support
):
    battery = coefficient_battery(
        cfg,
        template.descriptor_ids,
        template.families,
        template_key=template.name,
        stream="smoke",
    )
    law, syn = make_laws(battery, h=1, theta=0, regime="additive")
    data = hierarchical_draws(cfg, template, law, syn, stream="smoke", replicate=0)
    accumulator = PatientMoments(
        template,
        data["real_unit_index"],
        support.real_available,
        support.reference_available,
    )
    stats = accumulator.integrate(data["reference"], data["real"])
    counts = np.array([[3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0]])
    actual = accumulator.pool(stats, counts)["ccc"][0, 0]
    assert np.isfinite(actual)
    rows = np.concatenate(
        [
            np.flatnonzero(data["real_unit_index"] == unit)
            for unit in np.repeat(np.arange(11), counts[0])
        ]
    )
    expected = weighted_joint_moments(
        data["reference"][data["real_unit_index"][rows], 0],
        data["real"][rows, 0],
        np.ones(len(rows)) / len(rows),
    )
    np.testing.assert_allclose(actual, expected.ccc, atol=1e-12)


def test_degenerate_reference_bootstrap_is_undefined_even_with_complete_original_support(
    template, support
):
    unit_index = np.repeat(np.arange(11), 2)
    reference = np.repeat(np.arange(11, dtype=float)[:, None], 8, axis=1)
    alternate = reference[unit_index] + np.tile(np.array([0.0, 2.0])[:, None], (11, 8))
    accumulator = PatientMoments(
        template, unit_index, support.real_available, support.reference_available
    )
    moments = accumulator.integrate(reference, alternate)
    assert np.isfinite(accumulator.pool(moments)["ccc"]).all()
    counts = np.zeros((1, 11), dtype=int)
    counts[0, 3] = 11
    degenerate = accumulator.pool(moments, counts)
    assert np.equal(degenerate["reference_variance"], 0).all()
    assert np.greater(degenerate["disagreement_variance"], 0).all()
    assert np.isnan(degenerate["ccc"]).all()


def test_support_requires_explicit_masks(template):
    with pytest.raises(ValueError, match="Boolean"):
        SimulationSupport(np.ones((11, 8)), np.ones((22, 8))).validate(template)
    with pytest.raises(ValueError, match="Boolean"):
        SimulationSupport(np.ones((11, 8), bool), np.ones((22, 8), bool)).validate(
            template, require_synthetic=True
        )


def test_finite_ranges_resume_identically_and_retain_failed_descriptors(
    cfg, template, support
):
    support.real_available[:, 0] = False
    combined = finite_sample_tables(
        cfg,
        template,
        support,
        regime="additive",
        theta=0,
        start=0,
        stop=3,
        stream="smoke",
    )["finite_sample_mechanism"]
    resumed = finite_sample_tables(
        cfg,
        template,
        support,
        regime="additive",
        theta=0,
        start=2,
        stop=3,
        stream="smoke",
    )["finite_sample_mechanism"]
    pd.testing.assert_frame_equal(combined.iloc[2:].reset_index(drop=True), resumed)
    assert (combined.real_undefined_fraction == 1 / 8).all()
    assert (combined.evaluable_descriptor_count == 7).all()


def test_coverage_fixed_null_and_all_signed_comparisons_are_retained(
    cfg, template, support
):
    cfg["simulation"]["bootstrap_replicates_per_interval_audit_dataset"] = 39
    tables = coverage_tables(
        cfg,
        template,
        support,
        regime="heteroscedastic",
        start=0,
        stop=3,
        stream="smoke",
    )
    values = tables["coverage_repetitions"]
    null = values[values.endpoint == "contrast:second_minus_first"]
    assert (null.population_truth == 0).all()
    assert (
        values[values.endpoint == "contrast:higher_minus_first"].population_truth > 0
    ).all()
    assert (
        values[values.endpoint == "contrast:lower_minus_first"].population_truth < 0
    ).all()
    assert len(values) == 3 * (8 + 4 + 3)
    summary = summarize_coverage(values)
    assert len(summary) == 15
    assert summary.coverage_all_datasets.between(0, 1).all()
    assert (summary.datasets == 3).all()
    with pytest.raises(ValueError, match="overlap"):
        summarize_coverage(pd.concat([values, values]))


def test_failed_selected_feature_never_replaced_and_coverage_counts_failure(
    cfg, template, support
):
    cfg["simulation"]["bootstrap_replicates_per_interval_audit_dataset"] = 19
    support.real_available[:, 0] = False
    values = coverage_tables(
        cfg, template, support, regime="additive", start=0, stop=1, stream="smoke"
    )["coverage_repetitions"]
    first = values[values.endpoint == "utility:first"].iloc[0]
    assert not first.interval_issued
    assert not first.covered
    assert np.isnan(first.point_estimate)
    summary = summarize_coverage(values)
    first_summary = summary[summary.endpoint == "utility:first"].iloc[0]
    assert first_summary.coverage_all_datasets == 0
    assert first_summary.interval_failure_fraction == 1
