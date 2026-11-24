"""Ranks must be deterministic and evaluation cannot rescue missing selections."""

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from redesign2.moments import weighted_joint_moments, weighted_reference_moments
from redesign2.policies import (
    construct_screens,
    evaluate_selection,
    family_composition,
    fixed_budget,
    selection_overlap,
    utility_contrasts,
)
from redesign2.transport import fit_source_model, predict_target


def screens():
    ids = tuple("abcdefgh")
    return construct_screens(
        source_ccc=dict(zip(ids, [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])),
        synthetic_ccc=dict(zip(ids, [0.4, 0.9, 0.8, 0.3, 0.2, 0.1, 0.0, -0.1])),
        error_transport=dict(zip(ids, [0.4, 0.3, 0.9, 0.8, 0.2, 0.1, 0.0, -0.1])),
    )


@pytest.mark.parametrize(
    "size,budget", [(1, 1), (3, 1), (4, 1), (7, 1), (8, 2), (43, 10), (192, 48)]
)
def test_universal_budget(size, budget):
    assert fixed_budget(size) == budget


def test_stable_identity_ties_and_immutable_selection():
    scores = dict.fromkeys(reversed("abcdefgh"), 0.5)
    selections = construct_screens(
        source_ccc=scores, synthetic_ccc=scores, error_transport=scores
    )
    assert all(item.selected_ids == ("a", "b") for item in selections)
    with pytest.raises(FrozenInstanceError):
        selections[0].selected_ids = ("g", "h")
    scores["h"] = 0.9
    assert selections[0].ranked_scores == (0.5,) * 8


def test_all_procedures_require_same_audited_defined_universe():
    with pytest.raises(ValueError, match="same construction"):
        construct_screens(
            source_ccc={"a": 0.5},
            synthetic_ccc={"a": 0.5, "b": 0.5},
            error_transport={"a": 0.5},
        )
    with pytest.raises(ValueError, match="defined"):
        construct_screens(
            source_ccc={"a": np.nan},
            synthetic_ccc={"a": 0.5},
            error_transport={"a": 0.5},
        )
    assert set(inspect.signature(construct_screens).parameters) == {
        "source_ccc",
        "synthetic_ccc",
        "error_transport",
    }


def test_utilities_required_contrast_directions_regret_and_chance():
    selections = screens()
    observed = dict(zip("abcdefgh", [0.2, 0.4, 0.8, 0.6, 0.5, 0.3, 0.1, -0.2]))
    evaluations = [evaluate_selection(item, observed) for item in selections]
    contrasts = utility_contrasts(evaluations)
    assert [item.utility for item in evaluations] == pytest.approx([0.3, 0.6, 0.7])
    assert [item["difference"] for item in contrasts] == pytest.approx([0.4, -0.1, 0.3])
    assert [item["primary"] for item in contrasts] == [True, True, False]
    assert evaluations[0].observed_criterion_regret == pytest.approx(0.4)
    assert evaluations[1].top_quartile_precision == 0.5
    assert evaluations[2].top_quartile_precision == 1
    assert all(item.chance_precision == 0.25 for item in evaluations)
    assert selection_overlap(selections[0], selections[1]) == 0.5
    assert family_composition(
        selections[0], dict(zip("abcdefgh", ["shape", "size"] * 4))
    ) == {"shape": 1, "size": 1}


def test_any_missing_selected_criterion_invalidates_utility_without_replacement():
    selection = screens()[0]
    observed = dict.fromkeys("abcdefgh", 0.7)
    observed["b"] = np.nan
    result = evaluate_selection(selection, observed)
    assert np.isnan(result.utility)
    assert result.unevaluable_selected_ids == ("b",)
    assert selection.selected_ids == ("a", "b")
    assert np.isnan(result.observed_criterion_regret)
    assert np.isnan(result.top_quartile_precision)


def test_missing_unselected_criterion_preserves_utility_but_not_topk_or_regret():
    selection = screens()[0]
    observed = dict.fromkeys("abcdefg", 0.7)
    result = evaluate_selection(selection, observed)
    assert result.utility == 0.7
    assert result.utility_reason is None
    assert result.secondary_reason == "incomplete_observed_target_battery"
    assert np.isnan(result.observed_criterion_regret)
    assert np.isnan(result.top_quartile_precision)


def test_every_primary_selection_survives_independent_descriptor_units_and_shifts():
    x = np.repeat([-2, -1, 0, 1, 2], 2).astype(float)
    signs = np.tile([-1, 1], 5)
    weights = [0.1] * 10

    def construct(scales, shifts):
        source, synth, transport = {}, {}, {}
        for index, (scale, shift) in enumerate(zip(scales, shifts)):
            descriptor = f"d{index:02d}"
            xt = (0.8 + 0.1 * index) * x + 0.2 * index
            wr = x + 0.1 * index + (0.2 + 0.1 * index) * signs
            ws = xt + 0.05 * index + (1.1 - 0.08 * index) * signs
            source_moments = weighted_joint_moments(
                scale * x + shift, scale * wr + shift, weights
            )
            source[descriptor] = source_moments.ccc
            synth[descriptor] = weighted_joint_moments(
                scale * xt + shift, scale * ws + shift, weights
            ).ccc
            transport[descriptor] = predict_target(
                fit_source_model(source_moments, independent_patients=10),
                weighted_reference_moments(scale * xt + shift, weights),
            ).ccc
        return construct_screens(
            source_ccc=source, synthetic_ccc=synth, error_transport=transport
        )

    original = construct(np.ones(8), np.zeros(8))
    converted = construct(
        np.array([1e-7, -0.01, 0.1, 3, -20, 1e2, 1e4, 1e7]),
        np.array([0, 0.03, 2, -4, 8, 200, -1e5, 3e7]),
    )
    assert [item.selected_ids for item in converted] == [
        item.selected_ids for item in original
    ]
    for before, after in zip(original, converted):
        assert after.ranked_scores == pytest.approx(before.ranked_scores, abs=1e-13)
