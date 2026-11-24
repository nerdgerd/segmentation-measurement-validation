"""Construct fixed-budget screens from permitted scalar scores only.

No loader, evaluation profile, or alternate contour enters the construction
interface. Evaluation consumes a completed immutable screen and cannot replace
an unevaluable selected descriptor.
"""

from collections import Counter
from dataclasses import dataclass
from math import isfinite, nan
from typing import Mapping, Sequence


PRIMARY_POLICIES = ("source_ccc", "synthetic_ccc", "error_transport")


@dataclass(frozen=True)
class PolicySelection:
    policy: str
    universe: tuple[str, ...]
    ranked_ids: tuple[str, ...]
    ranked_scores: tuple[float, ...]
    selected_ids: tuple[str, ...]
    budget: int


@dataclass(frozen=True)
class SelectionEvaluation:
    policy: str
    utility: float
    utility_reason: str | None
    unevaluable_selected_ids: tuple[str, ...]
    top_quartile_precision: float
    observed_criterion_regret: float
    secondary_reason: str | None
    criterion_top_ids: tuple[str, ...]
    chance_precision: float


def fixed_budget(descriptor_count: int) -> int:
    if (
        not isinstance(descriptor_count, int)
        or isinstance(descriptor_count, bool)
        or descriptor_count < 1
    ):
        raise ValueError("A nonempty construction universe is required")
    return max(1, descriptor_count // 4)


def construct_screens(
    *,
    source_ccc: Mapping[str, float],
    synthetic_ccc: Mapping[str, float],
    error_transport: Mapping[str, float],
) -> tuple[PolicySelection, ...]:
    """Rank the same already audited universe for all three primary procedures.

    An undefined score is an unresolved construction eligibility condition, never a
    low score or a reason to borrow another procedure's ranking.
    """
    universe = tuple(sorted(source_ccc))
    if any(
        not isinstance(identifier, str) or not identifier for identifier in universe
    ):
        raise ValueError("Stable descriptor identifiers must be nonempty strings")
    budget = fixed_budget(len(universe))
    selections = []
    for policy, scores in zip(
        PRIMARY_POLICIES, (source_ccc, synthetic_ccc, error_transport)
    ):
        if set(scores) != set(universe):
            raise ValueError(
                "All three policies require the same construction descriptor identities"
            )
        if any(not isfinite(score) for score in scores.values()):
            raise ValueError("All construction scores must be defined before ranking")
        ranked = tuple(
            sorted(universe, key=lambda identifier: (-scores[identifier], identifier))
        )
        selections.append(
            PolicySelection(
                policy,
                universe,
                ranked,
                tuple(float(scores[item]) for item in ranked),
                ranked[:budget],
                budget,
            )
        )
    return tuple(selections)


def evaluate_selection(
    selection: PolicySelection, observed_target_ccc: Mapping[str, float | None]
) -> SelectionEvaluation:
    """Continuous utility uses the entire locked selection or is undefined."""
    if set(observed_target_ccc) - set(selection.universe):
        raise ValueError("Evaluation descriptors outside the locked universe")

    def evaluable(identifier: str) -> bool:
        value = observed_target_ccc.get(identifier)
        return value is not None and isfinite(value)

    missing_selected = tuple(
        identifier for identifier in selection.selected_ids if not evaluable(identifier)
    )
    utility = (
        nan
        if missing_selected
        else sum(float(observed_target_ccc[item]) for item in selection.selected_ids)
        / selection.budget
    )
    utility_reason = "unevaluable_selected_descriptor" if missing_selected else None
    complete = all(evaluable(identifier) for identifier in selection.universe)
    if complete:
        top = tuple(
            sorted(
                selection.universe,
                key=lambda item: (-float(observed_target_ccc[item]), item),
            )[: selection.budget]
        )
        precision = len(set(selection.selected_ids) & set(top)) / selection.budget
        regret = (
            sum(float(observed_target_ccc[item]) for item in top) / selection.budget
            - utility
        )
        secondary_reason = None
    else:
        top, precision, regret = (), nan, nan
        secondary_reason = "incomplete_observed_target_battery"
    return SelectionEvaluation(
        selection.policy,
        utility,
        utility_reason,
        missing_selected,
        precision,
        regret,
        secondary_reason,
        top,
        selection.budget / len(selection.universe),
    )


def utility_contrasts(
    evaluations: Sequence[SelectionEvaluation],
) -> tuple[dict[str, object], ...]:
    by_policy = {item.policy: item for item in evaluations}
    if len(by_policy) != len(evaluations) or set(by_policy) != set(PRIMARY_POLICIES):
        raise ValueError("Exactly one evaluation for each primary policy is required")
    contrasts = []
    for first, second, primary in (
        ("error_transport", "source_ccc", True),
        ("synthetic_ccc", "error_transport", True),
        ("synthetic_ccc", "source_ccc", False),
    ):
        defined = all(by_policy[key].utility_reason is None for key in (first, second))
        contrasts.append(
            {
                "first": first,
                "second": second,
                "primary": primary,
                "difference": by_policy[first].utility - by_policy[second].utility
                if defined
                else nan,
                "reason": None if defined else "unevaluable_policy_utility",
            }
        )
    return tuple(contrasts)


def selection_overlap(first: PolicySelection, second: PolicySelection) -> float:
    if first.universe != second.universe or first.budget != second.budget:
        raise ValueError("Selection overlap requires the same universe and budget")
    return len(set(first.selected_ids) & set(second.selected_ids)) / first.budget


def family_composition(
    selection: PolicySelection, descriptor_families: Mapping[str, str]
) -> dict[str, int]:
    if any(identifier not in descriptor_families for identifier in selection.universe):
        raise ValueError(
            "Descriptor-family mapping must be supplied for the locked universe"
        )
    return dict(
        sorted(
            Counter(
                descriptor_families[item] for item in selection.selected_ids
            ).items()
        )
    )
