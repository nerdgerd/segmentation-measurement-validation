"""Open target alternates only after every directional screen is committed.

Evaluation preserves the construction universe and selected identities. Matched
disagreement diagnostics get separately named support; they never replace a
construction score. Patient moment caches make the conditional bootstrap retain
reader/draw variability without repeatedly expanding the original draw table.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, fields, replace
import itertools
import json
from math import isfinite, nan
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .commitments import require_all_constructions
from .construction import (
    MatrixMeasurements,
    reference_matrix,
    source_pair_matrix,
    synthetic_matrix,
)
from .empirical_math import aggregate_patient_moments
from .moments import (
    JointMoments,
    ReferenceMoments,
    compare_disagreement,
    hierarchical_weights,
    summarize_disagreement,
)
from .policies import (
    PRIMARY_POLICIES,
    PolicySelection,
    construct_screens,
    evaluate_selection,
    family_composition,
    fixed_budget,
    selection_overlap,
    utility_contrasts,
)
from .provenance import RunWriter, content_hash, file_hash, inside, rng
from .transport import SourceModel, predict_target
from .uncertainty import (
    AUDIT_CONDITIONING,
    FULL_COHORT_CONDITIONING,
    conditional_bootstrap,
    summarize_bootstrap,
)
from .unit_audit import DIRECT_DESCRIPTOR_COUNT, audit_unit_conversions


def matched_diagnostic_matrices(
    real: MatrixMeasurements, synthetic: MatrixMeasurements
):
    """Match eligible reference units while preserving their real-reader/draw laws.

    A partly missing reader set does not multiply or remove synthetic rows. Both
    distributions condition on the same eligible units; reader/draw eligibility
    is retained within those units, and original coverage remains reportable.
    """
    if real.descriptor_ids != synthetic.descriptor_ids:
        raise ValueError("Diagnostic descriptor identities must be identical")
    units = sorted(set(map(tuple, real.keys[["patient_id", "unit_id"]].to_numpy())))
    if set(units) != set(
        map(tuple, synthetic.keys[["patient_id", "unit_id"]].to_numpy())
    ):
        raise ValueError("Diagnostic original unit support differs")
    lookup = {unit: index for index, unit in enumerate(units)}
    masks, indices = [], []
    for matrix in (real, synthetic):
        index = np.array(
            [
                lookup[tuple(row)]
                for row in matrix.keys[["patient_id", "unit_id"]].to_numpy()
            ]
        )
        present = np.zeros((len(units), len(real.descriptor_ids)), bool)
        np.logical_or.at(present, index, matrix.eligible)
        indices.append(index)
        masks.append(present)
    common = masks[0] & masks[1]
    return tuple(
        replace(matrix, eligible=matrix.eligible & common[index])
        for matrix, index in zip((real, synthetic), indices)
    )


def _profile_rows(profile, families):
    return [
        {
            "descriptor_id": descriptor,
            "family": families[descriptor],
            **asdict(moment),
            **{"support_" + key: value for key, value in asdict(support).items()},
        }
        for descriptor, moment, support in zip(
            profile.descriptor_ids, profile.moments, profile.support
        )
    ]


def _sufficient_moment_support(support, cfg):
    return (
        support.independent_patients
        >= cfg["features"]["min_evaluation_independent_patients"]
        and support.original_weighted_support
        >= cfg["features"]["min_evaluation_weighted_support"] - 1e-12
        and support.eligible_observations > 0
    )


def _comparisons(real_profile, synthetic_profile, cfg):
    undefined = JointMoments(
        nan, nan, nan, nan, nan, nan, nan, "insufficient_diagnostic_support"
    )
    comparisons, reasons = [], []
    for observed, synthetic, real_support, syn_support in zip(
        real_profile.moments,
        synthetic_profile.moments,
        real_profile.support,
        synthetic_profile.support,
    ):
        enough = _sufficient_moment_support(
            real_support, cfg
        ) and _sufficient_moment_support(syn_support, cfg)
        comparisons.append(
            compare_disagreement(
                synthetic if enough else undefined, observed if enough else undefined
            )
        )
        reasons.append(None if enough else "insufficient_matched_diagnostic_support")
    return comparisons, reasons


def _association(predicted, observed):
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    finite = np.isfinite(predicted) & np.isfinite(observed)
    count = int(finite.sum())
    complete = bool(finite.all())
    variable = (
        count >= 2 and np.ptp(predicted[finite]) > 0 and np.ptp(observed[finite]) > 0
    )
    available = (
        float(spearmanr(predicted[finite], observed[finite]).statistic)
        if variable
        else nan
    )
    mae = float(np.mean(np.abs(predicted[finite] - observed[finite]))) if count else nan
    reason = (
        "incomplete_observed_target_battery"
        if not complete
        else "constant_or_too_short_profile"
        if not variable
        else None
    )
    return {
        "universe_count": len(predicted),
        "evaluable_descriptors": count,
        "mean_absolute_ccc_error": mae if complete else nan,
        "profile_spearman": available if complete else nan,
        "available_descriptor_mae": mae,
        "available_descriptor_spearman": available,
        "prediction_error_reason": None
        if complete
        else "incomplete_observed_target_battery",
        "profile_reason": reason,
    }


def _one_selection(policy, scores, universe):
    if any(not isfinite(scores[descriptor]) for descriptor in universe):
        return None
    ranked = tuple(
        sorted(universe, key=lambda descriptor: (-scores[descriptor], descriptor))
    )
    budget = fixed_budget(len(universe))
    return PolicySelection(
        policy,
        tuple(universe),
        ranked,
        tuple(scores[descriptor] for descriptor in ranked),
        ranked[:budget],
        budget,
    )


def source_residual_diagnostics(
    reference, alternate, *, cohort, descriptor_ids, source_models, families
):
    """Describe the locked source projection's residuals; never fit another model."""
    matrix = source_pair_matrix(
        reference, alternate, cohort=cohort, descriptor_ids=descriptor_ids
    )
    rows = []
    for index, descriptor in enumerate(descriptor_ids):
        model = source_models[descriptor]
        support = hierarchical_weights(
            matrix.keys.patient_id.tolist(),
            matrix.keys.unit_id.tolist(),
            matrix.keys.observation_id.tolist(),
            eligible=matrix.eligible[:, index],
        )
        weights = np.asarray(support.weights)
        for position, key in enumerate(matrix.keys.itertuples()):
            eligible = bool(matrix.eligible[position, index])
            x, w = matrix.reference[position, index], matrix.alternate[position, index]
            residual = (
                w - x - model.mean_disagreement - model.slope * (x - model.source_mean)
                if eligible
                else nan
            )
            rows.append(
                {
                    "source_cohort": cohort,
                    "patient_id": key.patient_id,
                    "unit_id": key.unit_id,
                    "observation_id": key.observation_id,
                    "descriptor_id": descriptor,
                    "family": families[descriptor],
                    "reference_value": x,
                    "observed_alternate_value": w,
                    "observed_disagreement": w - x if eligible else nan,
                    "source_centered_reference": x - model.source_mean
                    if eligible
                    else nan,
                    "source_projection_residual": residual,
                    "hierarchical_weight": weights[position],
                    "eligible": eligible,
                    "original_weighted_support": support.original_support_coverage,
                    "model_refitted": False,
                }
            )
    return pd.DataFrame(rows)


def _statistic_values(evaluations, score_maps, observed, comparisons):
    values = {
        "utility:" + evaluation.policy: evaluation.utility for evaluation in evaluations
    }
    values.update(
        {
            "contrast:" + row["first"] + "-" + row["second"]: row["difference"]
            for row in utility_contrasts(evaluations)
        }
    )
    for policy in PRIMARY_POLICIES:
        profile = _association(
            list(score_maps[policy].values()),
            [observed[descriptor] for descriptor in score_maps[policy]],
        )
        values["ccc_mae:" + policy] = profile["mean_absolute_ccc_error"]
        values["profile_spearman:" + policy] = profile["profile_spearman"]
    disagreement = summarize_disagreement(comparisons)
    values["disagreement:median_signed_finite_log2_ratio"] = disagreement[
        "median_signed_finite_log2_ratio"
    ]
    values["disagreement:median_absolute_finite_log2_ratio"] = disagreement[
        "median_absolute_finite_log2_ratio"
    ]
    return values


def _bootstrap(
    cfg,
    *,
    scope,
    observed_cache,
    reference_cache,
    synthetic_cache,
    matched_real_cache,
    matched_synthetic_cache,
    selections,
    source_models,
    score_maps,
    source,
    target,
):
    patients = observed_cache.patient_ids
    for cache in (
        reference_cache,
        synthetic_cache,
        matched_real_cache,
        matched_synthetic_cache,
    ):
        if cache.patient_ids != patients:
            raise ValueError(
                "Shared target bootstrap must preserve exactly the same original patients"
            )
    threshold = cfg["features"]["min_evaluation_weighted_support"]
    original_observed = aggregate_patient_moments(
        observed_cache,
        min_independent_patients=cfg["features"]["min_evaluation_independent_patients"],
        min_weighted_support=threshold,
    )
    construction_coverage = cfg["features"]["min_construction_weighted_availability"]
    original_reference = aggregate_patient_moments(
        reference_cache,
        min_independent_patients=1,
        min_weighted_support=construction_coverage,
    )
    original_synthetic = aggregate_patient_moments(
        synthetic_cache,
        min_independent_patients=1,
        min_weighted_support=construction_coverage,
    )
    original_mreal = aggregate_patient_moments(
        matched_real_cache,
        min_independent_patients=cfg["features"]["min_evaluation_independent_patients"],
        min_weighted_support=threshold,
    )
    original_msyn = aggregate_patient_moments(
        matched_synthetic_cache,
        min_independent_patients=cfg["features"]["min_evaluation_independent_patients"],
        min_weighted_support=threshold,
    )
    universe = observed_cache.descriptor_ids
    frequencies = {policy: Counter() for policy in PRIMARY_POLICIES}
    selection_defined = Counter()
    failure_counts = Counter()
    original_selections = {selection.policy: selection for selection in selections}
    point_comparisons, _ = _comparisons(original_mreal, original_msyn, cfg)
    point_evaluations = [
        evaluate_selection(selection, original_observed.ccc_mapping())
        for selection in selections
    ]
    labels = tuple(
        _statistic_values(
            point_evaluations,
            score_maps,
            original_observed.ccc_mapping(),
            point_comparisons,
        )
    )

    def statistic(counts):
        observed = aggregate_patient_moments(
            observed_cache,
            counts,
            locked_support=original_observed.support,
            min_weighted_support=threshold,
        )
        maps = {policy: dict(scores) for policy, scores in score_maps.items()}
        if scope == "full":
            references = aggregate_patient_moments(
                reference_cache,
                counts,
                locked_support=original_reference.support,
                min_weighted_support=construction_coverage,
            )
            synthetic = aggregate_patient_moments(
                synthetic_cache,
                counts,
                locked_support=original_synthetic.support,
                min_weighted_support=construction_coverage,
            )
            maps["synthetic_ccc"] = synthetic.ccc_mapping()
            maps["error_transport"] = {
                descriptor: predict_target(
                    source_models[descriptor],
                    ReferenceMoments(moment.reference_mean, moment.reference_variance),
                ).ccc
                if support.evaluable
                else nan
                for descriptor, moment, support in zip(
                    universe, references.moments, references.support
                )
            }
        observed_map = observed.ccc_mapping()
        evaluations = []
        for policy in PRIMARY_POLICIES:
            selected = (
                _one_selection(policy, maps[policy], universe)
                if scope == "full"
                else original_selections[policy]
            )
            if selected is None:
                # Preserve the locked universe. Undefined construction in a
                # resample does not authorize a smaller battery or fallback.
                evaluation = evaluate_selection(original_selections[policy], {})
                evaluation = replace(
                    evaluation, utility_reason="undefined_bootstrap_construction_score"
                )
            else:
                evaluation = evaluate_selection(selected, observed_map)
                selection_defined[policy] += 1
                frequencies[policy].update(selected.selected_ids)
            if evaluation.utility_reason:
                failure_counts[(policy, evaluation.utility_reason)] += 1
            evaluations.append(evaluation)
        mreal = aggregate_patient_moments(
            matched_real_cache,
            counts,
            locked_support=original_mreal.support,
            min_weighted_support=threshold,
        )
        msyn = aggregate_patient_moments(
            matched_synthetic_cache,
            counts,
            locked_support=original_msyn.support,
            min_weighted_support=threshold,
        )
        # Patient-slot counts are not a new minimum-original-patients test.
        diagnostic_cfg = {
            **cfg,
            "features": {**cfg["features"], "min_evaluation_independent_patients": 1},
        }
        comparisons, _ = _comparisons(mreal, msyn, diagnostic_cfg)
        unavailable = JointMoments(nan, nan, nan, nan, nan, nan, nan)
        comparisons = [
            comparison
            if _sufficient_moment_support(real_support, cfg)
            and _sufficient_moment_support(syn_support, cfg)
            else compare_disagreement(unavailable, unavailable)
            for comparison, real_support, syn_support in zip(
                comparisons, original_mreal.support, original_msyn.support
            )
        ]
        return _statistic_values(evaluations, maps, observed_map, comparisons)

    distribution = conditional_bootstrap(
        patient_count=len(patients),
        statistic_ids=labels,
        statistic=statistic,
        rng=rng(cfg, "empirical_bootstrap", source, target, scope),
        conditioning=FULL_COHORT_CONDITIONING
        if scope == "full"
        else AUDIT_CONDITIONING,
        replicates=cfg["uncertainty"]["empirical_bootstrap_replicates"],
    )
    frequency_rows = [
        {
            "policy": policy,
            "descriptor_id": descriptor,
            "selected_replicates": frequencies[policy][descriptor],
            "defined_selection_replicates": selection_defined[policy],
            "all_replicates": distribution.replicates,
            "selection_frequency_all_replicates": frequencies[policy][descriptor]
            / distribution.replicates,
        }
        for policy in PRIMARY_POLICIES
        for descriptor in universe
    ]
    failures = [
        {"policy": policy, "reason": reason, "replicates": count}
        for (policy, reason), count in sorted(failure_counts.items())
    ]
    return (
        distribution,
        pd.DataFrame(frequency_rows),
        pd.DataFrame(failures, columns=["policy", "reason", "replicates"]),
    )


def evaluate_from_frames(
    cfg,
    *,
    source,
    target,
    scope,
    universe,
    selections,
    source_models,
    score_maps,
    target_reference,
    target_alternate,
    draw_features,
    draw_definedness,
    descriptor_families,
    coverage_nominal_allowed=False,
    locked_predictions=None,
):
    """Pure numerical evaluation used only after the file-level commitment gate.

    Artificial-data tests may call this directly. It has no loader and cannot
    acquire target measurements or change a construction commitment.
    """
    universe = tuple(universe)
    if (
        scope not in {"full", "audit"}
        or not universe
        or len(set(universe)) != len(universe)
    ):
        raise ValueError(
            "Evaluation requires a nonempty fixed universe and declared scope"
        )
    if set(descriptor_families) != set(universe) or any(
        not family for family in descriptor_families.values()
    ):
        raise ValueError("Every construction descriptor needs its audited family")
    if set(score_maps) != set(PRIMARY_POLICIES) or any(
        set(scores) != set(universe) for scores in score_maps.values()
    ):
        raise ValueError("Three complete locked construction score maps are required")
    if set(source_models) != set(universe):
        raise ValueError("The source model must cover the locked universe")
    by_policy = {selection.policy: selection for selection in selections}
    if set(by_policy) != set(PRIMARY_POLICIES) or any(
        selection.universe != universe for selection in selections
    ):
        raise ValueError("Selections must share the locked universe")
    reference = reference_matrix(
        target_reference, cohort=target, descriptor_ids=universe
    )
    observed = source_pair_matrix(
        target_reference, target_alternate, cohort=target, descriptor_ids=universe
    )
    synthetic = synthetic_matrix(
        reference, draw_features, draw_definedness, cohort=target
    )
    mreal, msyn = matched_diagnostic_matrices(observed, synthetic)
    caches = [
        matrix.patient_moments()
        for matrix in (observed, reference, synthetic, mreal, msyn)
    ]
    observed_cache, reference_cache, synthetic_cache, mreal_cache, msyn_cache = caches
    min_patients = cfg["features"]["min_evaluation_independent_patients"]
    threshold = cfg["features"]["min_evaluation_weighted_support"]
    (
        observed_profile,
        reference_profile,
        synthetic_profile,
        mreal_profile,
        msyn_profile,
    ) = [
        aggregate_patient_moments(
            cache, min_independent_patients=min_patients, min_weighted_support=threshold
        )
        for cache in caches
    ]
    observed_map = observed_profile.ccc_mapping()
    comparisons, diagnostic_reasons = _comparisons(mreal_profile, msyn_profile, cfg)
    comparison_rows = []
    for index, descriptor in enumerate(universe):
        comparison_rows.append(
            {
                "descriptor_id": descriptor,
                "family": descriptor_families[descriptor],
                "policy": "synthetic_ccc",
                "predicted_disagreement_mse": msyn_profile.moments[
                    index
                ].disagreement_mse,
                "observed_disagreement_mse": mreal_profile.moments[
                    index
                ].disagreement_mse,
                "predicted_bias": msyn_profile.moments[index].disagreement_mean,
                "observed_bias": mreal_profile.moments[index].disagreement_mean,
                **asdict(comparisons[index]),
                "diagnostic_reason": diagnostic_reasons[index],
                "observed_diagnostic_weighted_support": mreal_profile.support[
                    index
                ].original_weighted_support,
                "synthetic_diagnostic_weighted_support": msyn_profile.support[
                    index
                ].original_weighted_support,
                "observed_primary_ccc": observed_map[descriptor],
                "matched_observed_ccc": mreal_profile.moments[index].ccc,
                "matched_synthetic_ccc": msyn_profile.moments[index].ccc,
                "prediction_origin": "matched_support_synthetic_diagnostic",
                "synthetic_construction_score": score_maps["synthetic_ccc"][descriptor],
            }
        )
    family_summaries = []
    for family in ["all", *sorted(set(descriptor_families.values()))]:
        group = [
            comparison
            for descriptor, comparison in zip(universe, comparisons)
            if family == "all" or descriptor_families[descriptor] == family
        ]
        family_summaries.append({"family": family, **summarize_disagreement(group)})
    transport_comparisons, transport_rows = [], []
    undefined_moments = JointMoments(nan, nan, nan, nan, nan, nan, nan)
    for index, descriptor in enumerate(universe):
        if locked_predictions is not None:
            predicted = locked_predictions[descriptor]
        elif scope == "full":
            reference_moment = reference_profile.moments[index]
            predicted = predict_target(
                source_models[descriptor],
                ReferenceMoments(
                    reference_moment.reference_mean, reference_moment.reference_variance
                ),
            )
        else:
            predicted = undefined_moments
        observed_moment = observed_profile.moments[index]
        enough = _sufficient_moment_support(observed_profile.support[index], cfg)
        comparison = compare_disagreement(
            predicted, observed_moment if enough else undefined_moments
        )
        transport_comparisons.append(comparison)
        transport_rows.append(
            {
                "descriptor_id": descriptor,
                "family": descriptor_families[descriptor],
                "policy": "error_transport",
                "predicted_disagreement_mse": predicted.disagreement_mse,
                "observed_disagreement_mse": observed_moment.disagreement_mse,
                "predicted_bias": predicted.disagreement_mean,
                "observed_bias": observed_moment.disagreement_mean,
                "log2_predicted_to_observed_rmse_ratio": comparison.log2_rmse_ratio,
                "rmse_stratum": comparison.rmse_stratum.replace(
                    "synthetic", "predicted"
                ),
                "standardized_bias_difference": comparison.standardized_bias_difference,
                "predicted_reference_slope": comparison.synthetic_reference_slope,
                "observed_reference_slope": comparison.observed_reference_slope,
                "reference_slope_difference": comparison.reference_slope_difference,
                "prediction_origin": "locked_discovery_prediction"
                if scope == "audit"
                else "locked_full_prediction",
                "support": "construction_prediction_compared_with_observed_evaluation_support_without_refitting",
                "reason": None
                if enough and isfinite(predicted.disagreement_mse)
                else "insufficient_support_or_unavailable_locked_prediction",
            }
        )
    transport_family_rows = []
    for family in ["all", *sorted(set(descriptor_families.values()))]:
        group = [
            comparison
            for descriptor, comparison in zip(universe, transport_comparisons)
            if family == "all" or descriptor_families[descriptor] == family
        ]
        summary = summarize_disagreement(group)
        summary = {
            key.replace("synthetic", "predicted"): value
            for key, value in summary.items()
        }
        transport_family_rows.append(
            {"family": family, "policy": "error_transport", **summary}
        )
    evaluations = [
        evaluate_selection(by_policy[policy], observed_map)
        for policy in PRIMARY_POLICIES
    ]
    policy_profiles, descriptor_predictions = [], []
    for policy in PRIMARY_POLICIES:
        policy_profiles.append(
            {
                "policy": policy,
                **_association(
                    [score_maps[policy][descriptor] for descriptor in universe],
                    [observed_map[descriptor] for descriptor in universe],
                ),
                "score_origin": "locked_discovery_construction"
                if scope == "audit"
                else "locked_full_construction",
            }
        )
        for descriptor in universe:
            prediction = score_maps[policy][descriptor]
            observed_ccc = observed_map[descriptor]
            descriptor_predictions.append(
                {
                    "policy": policy,
                    "descriptor_id": descriptor,
                    "family": descriptor_families[descriptor],
                    "construction_score": prediction,
                    "observed_target_ccc": observed_ccc,
                    "absolute_ccc_error": abs(prediction - observed_ccc)
                    if isfinite(observed_ccc)
                    else nan,
                }
            )
    secondary = []
    for evaluation in evaluations:
        secondary.append(
            {
                **asdict(evaluation),
                "budget": by_policy[evaluation.policy].budget,
                "universe_count": len(universe),
                "selected_ids": json.dumps(by_policy[evaluation.policy].selected_ids),
                "family_composition": json.dumps(
                    family_composition(
                        by_policy[evaluation.policy], descriptor_families
                    ),
                    sort_keys=True,
                ),
            }
        )
    overlaps = [
        {
            "first": first,
            "second": second,
            "selected_set_overlap": selection_overlap(
                by_policy[first], by_policy[second]
            ),
        }
        for first, second in itertools.combinations(PRIMARY_POLICIES, 2)
    ]
    lofo = []
    for omitted in sorted(set(descriptor_families.values())):
        remaining = tuple(
            descriptor
            for descriptor in universe
            if descriptor_families[descriptor] != omitted
        )
        if not remaining:
            lofo.append(
                {
                    "omitted_family": omitted,
                    "universe_count": 0,
                    "budget": 0,
                    "reason": "empty_remaining_universe",
                }
            )
            continue
        screens = construct_screens(
            **{
                policy: {
                    descriptor: score_maps[policy][descriptor]
                    for descriptor in remaining
                }
                for policy in PRIMARY_POLICIES
            }
        )
        result = [
            evaluate_selection(
                screen,
                {descriptor: observed_map[descriptor] for descriptor in remaining},
            )
            for screen in screens
        ]
        for contrast in utility_contrasts(result):
            lofo.append(
                {
                    "omitted_family": omitted,
                    "universe_count": len(remaining),
                    "budget": fixed_budget(len(remaining)),
                    **contrast,
                    **{
                        "utility_" + evaluation.policy: evaluation.utility
                        for evaluation in result
                    },
                    **{
                        "selected_" + screen.policy: json.dumps(screen.selected_ids)
                        for screen in screens
                    },
                }
            )
    distribution, frequencies, bootstrap_failures = _bootstrap(
        cfg,
        scope=scope,
        observed_cache=observed_cache,
        reference_cache=reference_cache,
        synthetic_cache=synthetic_cache,
        matched_real_cache=mreal_cache,
        matched_synthetic_cache=msyn_cache,
        selections=selections,
        source_models=source_models,
        score_maps=score_maps,
        source=source,
        target=target,
    )
    summaries = pd.DataFrame(
        summarize_bootstrap(
            distribution,
            nominal_interval_allowed=coverage_nominal_allowed,
            nominal_level=cfg["uncertainty"]["nominal_pointwise_level"],
            max_undefined_fraction=cfg["uncertainty"][
                "max_undefined_bootstrap_fraction_for_nominal_interval"
            ],
            point_estimates=_statistic_values(
                evaluations, score_maps, observed_map, comparisons
            ),
        )
    )
    # Coverage auditing targets criterion utility/contrasts, not profile correlation
    # or every disagreement diagnostic. Those remain stability descriptions.
    auxiliary = ~summaries.statistic_id.str.startswith(
        ("utility:", "contrast:")
    ) & summaries.interval_kind.ne("not_issued")
    summaries.loc[auxiliary, "interval_kind"] = "bootstrap_sampling_stability_range"
    replicate_table = pd.DataFrame(
        distribution.values, columns=distribution.statistic_ids
    )
    replicate_table.insert(0, "replicate", np.arange(distribution.replicates))
    failures = [
        {
            "descriptor_id": descriptor,
            "failure_stage": "observed_evaluation",
            "reason": support.reason,
        }
        for descriptor, support in zip(universe, observed_profile.support)
        if support.reason
    ]
    failures.extend(
        {
            "descriptor_id": descriptor,
            "failure_stage": "matched_diagnostic",
            "reason": reason,
        }
        for descriptor, reason in zip(universe, diagnostic_reasons)
        if reason
    )
    tables = {
        "observed_moments": pd.DataFrame(
            _profile_rows(observed_profile, descriptor_families)
        ),
        "reference_evaluation_moments": pd.DataFrame(
            _profile_rows(reference_profile, descriptor_families)
        ),
        "synthetic_evaluation_moments": pd.DataFrame(
            _profile_rows(synthetic_profile, descriptor_families)
        ),
        "matched_observed_moments": pd.DataFrame(
            _profile_rows(mreal_profile, descriptor_families)
        ),
        "matched_synthetic_moments": pd.DataFrame(
            _profile_rows(msyn_profile, descriptor_families)
        ),
        "descriptor_disagreement": pd.DataFrame(comparison_rows),
        "disagreement_family_summary": pd.DataFrame(family_summaries),
        "transport_disagreement": pd.DataFrame(transport_rows),
        "transport_disagreement_family_summary": pd.DataFrame(transport_family_rows),
        "profile_agreement": pd.DataFrame(policy_profiles),
        "descriptor_prediction_errors": pd.DataFrame(descriptor_predictions),
        "utility_contrasts": pd.DataFrame(utility_contrasts(evaluations)),
        "secondary_selection_metrics": pd.DataFrame(secondary),
        "selection_overlap": pd.DataFrame(overlaps),
        "leave_one_family_out": pd.DataFrame(lofo),
        "bootstrap_replicates": replicate_table,
        "bootstrap_summary": summaries,
        "bootstrap_selection_frequency": frequencies,
        "bootstrap_failures": bootstrap_failures,
        "evaluation_failures": pd.DataFrame(
            failures, columns=["descriptor_id", "failure_stage", "reason"]
        ),
    }
    for table in tables.values():
        table["source"], table["target"], table["scope"] = source, target, scope
        if "universe_count" not in table:
            table["universe_count"] = len(universe)
        table["full_construction_universe_count"], table["patient_count"] = (
            len(universe),
            len(observed_cache.patient_ids),
        )
        table["conditioning"] = distribution.conditioning
    return tables


def _run_manifest(root, cfg, expected_stage=None):
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["protocol_sha256"] != content_hash(cfg)
        or manifest["status"] != "complete"
    ):
        raise ValueError("Evaluation input run is incomplete or uses another protocol")
    if expected_stage and manifest["stage"] != expected_stage:
        raise ValueError("Evaluation input has an unexpected stage")
    return manifest


def _read_output(root, manifest, name, paths, *, json_data=False):
    path = inside(root, root / name)
    if name not in manifest["outputs"] or file_hash(path) != manifest["outputs"][name]:
        raise ValueError(f"Evaluation input missing or changed: {name}")
    paths.append(path)
    if json_data:
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(
        path,
        float_precision="round_trip",
        dtype={
            "cohort": str,
            "patient_id": str,
            "unit_id": str,
            "annotation_id": str,
            "descriptor_id": str,
            "policy": str,
        },
        keep_default_na=True,
    )


def _joint_mapping(frame, universe):
    names = [field.name for field in fields(JointMoments)]
    result = {}
    for row in frame.loc[frame.descriptor_id.isin(universe)].to_dict("records"):
        values = {name: row[name] for name in names}
        values["ccc_reason"] = (
            None
            if pd.isna(values["ccc_reason"]) or values["ccc_reason"] == ""
            else values["ccc_reason"]
        )
        if row["descriptor_id"] in result:
            raise ValueError("Repeated descriptor in a moment artifact")
        result[row["descriptor_id"]] = JointMoments(**values)
    if set(result) != set(universe):
        raise ValueError("Moment artifact does not cover the committed universe")
    return result


def _coverage_nominal_permission(root, cfg, prepared_run, paths):
    """A coverage decision must share the exact prepared-input lineage it qualifies."""
    from .provenance import verify_run

    meta = _run_manifest(root, cfg, "simulation_coverage")
    errors = verify_run(root)
    if errors:
        raise ValueError(f"Coverage artifact failed integrity verification: {errors}")
    prepared_manifest = prepared_run / "run_manifest.json"
    if meta["inputs"].get(str(prepared_manifest)) != file_hash(prepared_manifest):
        raise ValueError(
            "Coverage artifact does not use the evaluation's prepared input run"
        )
    decision = _read_output(root, meta, "coverage_decision.json", paths, json_data=True)
    paths.append(root / "run_manifest.json")
    for key in ["nominal_audit_intervals_allowed", "required_cells_complete"]:
        if not isinstance(decision.get(key), bool):
            raise ValueError(
                "Coverage permission and completeness must be explicit JSON Booleans"
            )
    shortfalls = decision.get("shortfall_count")
    if (
        not isinstance(shortfalls, int)
        or isinstance(shortfalls, bool)
        or shortfalls < 0
    ):
        raise ValueError(
            "Coverage decision requires a nonnegative integer shortfall count"
        )
    if decision.get("source_table") != "coverage_summary.csv":
        raise ValueError(
            "Coverage decision must name its declared summary source table"
        )
    summary = _read_output(root, meta, "coverage_summary.csv", paths)
    if (
        "material_coverage_shortfall" not in summary
        or not summary.material_coverage_shortfall.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all()
    ):
        raise ValueError("Coverage summary requires explicit Boolean shortfall flags")
    if int(summary.material_coverage_shortfall.sum()) != shortfalls:
        raise ValueError("Coverage decision disagrees with its source summary")
    if decision["nominal_audit_intervals_allowed"] and (
        summary.empty or not decision["required_cells_complete"] or shortfalls
    ):
        raise ValueError(
            "Coverage decision grants nominal intervals despite a required audit failure"
        )
    return decision["nominal_audit_intervals_allowed"]


def evaluate_direction(
    workspace,
    cfg,
    prepared_run,
    generation_run,
    construction_run,
    all_commitment_paths,
    source,
    target,
    scope,
    run_id,
    coverage_nominal_allowed=False,
    coverage_run=None,
):
    """Evaluate one direction only after all eight immutable constructions verify."""
    # This is intentionally before any target file/header is opened.
    commitments = require_all_constructions(cfg, all_commitment_paths)
    if (
        scope not in {"full", "audit"}
        or {"source": source, "target": target} not in cfg["directions"]
    ):
        raise ValueError("Undeclared evaluation scope/direction")
    construction_scope = "full" if scope == "full" else "discovery"
    direction = f"{source}_to_{target}"
    committed = next(
        entry
        for entry in commitments
        if entry["direction"] == direction and entry["scope"] == construction_scope
    )
    workspace, prepared_run, generation_run, construction_run = map(
        lambda value: Path(value).resolve(),
        (workspace, prepared_run, generation_run, construction_run),
    )
    from .prepare_inputs import require_protocol_lock

    if content_hash(require_protocol_lock(workspace)) != content_hash(cfg):
        raise ValueError(
            "Evaluation configuration differs from the current execution lock"
        )
    paths = [Path(path).resolve() for path in all_commitment_paths]
    paths.append(workspace / "config/protocol.lock.yaml")
    if coverage_run is not None:
        coverage_run = Path(coverage_run).resolve()
        coverage_nominal_allowed = _coverage_nominal_permission(
            coverage_run, cfg, prepared_run, paths
        )
    elif coverage_nominal_allowed:
        raise ValueError(
            "Nominal audit intervals require their verified coverage artifact"
        )
    pmeta = _run_manifest(prepared_run, cfg, "prepare_inputs")
    gmeta = _run_manifest(generation_run, cfg)
    cmeta = _run_manifest(construction_run, cfg, "construction")
    if committed["artifacts"].get(
        str(construction_run / "run_manifest.json")
    ) != file_hash(construction_run / "run_manifest.json"):
        raise ValueError("Supplied construction run is not the committed artifact")
    paths.extend(
        root / "run_manifest.json"
        for root in (prepared_run, generation_run, construction_run)
    )
    if (
        gmeta.get("extra", {}).get("source") != source
        or gmeta.get("extra", {}).get("target") != target
    ):
        raise PermissionError(
            "Synthetic input direction differs from committed construction"
        )
    if (
        cmeta.get("extra", {}).get("direction") != direction
        or cmeta.get("extra", {}).get("scope") != construction_scope
    ):
        raise PermissionError("Construction input belongs to another direction/scope")
    for root in (prepared_run, generation_run):
        path = root / "run_manifest.json"
        if cmeta["inputs"].get(str(path)) != file_hash(path):
            raise ValueError(
                "Evaluation preparation/generation differs from the committed construction inputs"
            )
    status = _read_output(
        construction_run, cmeta, "construction_status.json", paths, json_data=True
    )
    universe = tuple(status["universe"])
    if not status["ready_for_commitment"] or list(universe) != committed["universe"]:
        raise ValueError("Evaluation universe differs from its immutable commitment")
    score_table = _read_output(
        construction_run, cmeta, "construction_scores.csv", paths
    )
    score_table = score_table.loc[score_table.descriptor_id.isin(universe)]
    score_maps = {
        policy: dict(zip(group.descriptor_id, group.score))
        for policy, group in score_table.groupby("policy")
    }
    selections = construct_screens(**score_maps)
    selection_table = _read_output(construction_run, cmeta, "selections.csv", paths)
    for selection in selections:
        ids = (
            selection_table.loc[selection_table.policy.eq(selection.policy)]
            .sort_values("rank")
            .descriptor_id.tolist()
        )
        if (
            ids != list(selection.selected_ids)
            or ids != committed["selections"][selection.policy]
        ):
            raise ValueError(
                "Selection differs from its immutable construction commitment"
            )
    models = _read_output(construction_run, cmeta, "source_models.csv", paths)
    model_names = {field.name for field in fields(SourceModel)}
    source_models = {}
    for row in models.loc[models.descriptor_id.isin(universe)].to_dict("records"):
        values = {name: row[name] for name in model_names}
        values["reason"] = (
            None
            if pd.isna(values["reason"]) or values["reason"] == ""
            else values["reason"]
        )
        source_models[row["descriptor_id"]] = SourceModel(**values)
    predictions = _read_output(construction_run, cmeta, "target_predictions.csv", paths)
    locked_predictions = {}
    moment_names = [
        field.name for field in fields(JointMoments) if field.name != "ccc_reason"
    ]
    for row in predictions.loc[predictions.descriptor_id.isin(universe)].to_dict(
        "records"
    ):
        reason = (
            None if pd.isna(row["reason"]) or row["reason"] == "" else row["reason"]
        )
        locked_predictions[row["descriptor_id"]] = JointMoments(
            **{name: row[name] for name in moment_names}, ccc_reason=reason
        )
    if set(locked_predictions) != set(universe):
        raise ValueError(
            "Locked disagreement predictions do not cover the committed universe"
        )
    descriptor_path = workspace / cfg["workspace"]["descriptor_manifest"]
    paths.append(descriptor_path)
    metadata = pd.read_csv(descriptor_path, dtype=str, keep_default_na=False)
    metadata = metadata.loc[
        metadata.domain.eq(cfg["cohorts"][target]["modality"])
        & metadata.feature_id.isin(universe)
    ]
    families = dict(zip(metadata.feature_id, metadata.family))
    split_path = workspace / cfg["resolved_metadata"]["patient_split"]
    paths.append(split_path)
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    target_patients = split.loc[split.cohort.eq(target)]
    allowed_patients = (
        set(target_patients.loc[target_patients.assignment.eq("audit"), "patient_id"])
        if scope == "audit"
        else set(target_patients.patient_id)
    )
    all_references = _read_output(
        prepared_run, pmeta, f"{target}/reference_features.csv", paths
    )
    # Every commitment, selection identity and input lineage has passed before this read.
    alternates = _read_output(
        prepared_run, pmeta, f"{target}/alternate_features.csv", paths
    )
    all_draws = _read_output(generation_run, gmeta, "draw_features.csv", paths)
    all_flags = _read_output(generation_run, gmeta, "draw_definedness.csv", paths)
    references = all_references.loc[
        all_references.patient_id.isin(allowed_patients)
        & all_references.descriptor_id.isin(universe)
    ]
    alternates = alternates.loc[
        alternates.patient_id.isin(allowed_patients)
        & alternates.descriptor_id.isin(universe)
    ]
    draws, flags = [
        frame.loc[frame.patient_id.isin(allowed_patients)]
        for frame in (all_draws, all_flags)
    ]
    if (
        set(references.patient_id) != allowed_patients
        or set(alternates.patient_id) != allowed_patients
    ):
        raise ValueError(
            "Evaluation patient set differs from the immutable target split"
        )
    tables = evaluate_from_frames(
        cfg,
        source=source,
        target=target,
        scope=scope,
        universe=universe,
        selections=selections,
        source_models=source_models,
        score_maps=score_maps,
        target_reference=references,
        target_alternate=alternates,
        draw_features=draws,
        draw_definedness=flags,
        descriptor_families=families,
        coverage_nominal_allowed=coverage_nominal_allowed,
        locked_predictions=locked_predictions,
    )
    source_reference = _read_output(
        prepared_run, pmeta, f"{source}/reference_features.csv", paths
    )
    source_alternate = _read_output(
        prepared_run, pmeta, f"{source}/alternate_features.csv", paths
    )
    residuals = source_residual_diagnostics(
        source_reference.loc[source_reference.descriptor_id.isin(universe)],
        source_alternate.loc[source_alternate.descriptor_id.isin(universe)],
        cohort=source,
        descriptor_ids=universe,
        source_models=source_models,
        families=families,
    )
    residuals["source"], residuals["target"], residuals["scope"] = source, target, scope
    tables["source_residuals"] = residuals
    source_joint_rows = []
    for row in models.loc[models.descriptor_id.isin(universe)].to_dict("records"):
        source_joint_rows.append(
            {
                "descriptor_id": row["descriptor_id"],
                "reference_mean": row["source_mean"],
                **{
                    field.name: row["source_" + field.name]
                    for field in fields(JointMoments)
                    if field.name != "reference_mean"
                },
            }
        )
    source_joint = _joint_mapping(pd.DataFrame(source_joint_rows), universe)
    construction_synthetic = _joint_mapping(
        _read_output(construction_run, cmeta, "synthetic_moments.csv", paths), universe
    )
    construction_patients = (
        set(
            target_patients.loc[
                target_patients.assignment.eq("discovery"), "patient_id"
            ]
        )
        if scope == "audit"
        else set(target_patients.patient_id)
    )
    construction_references = all_references.loc[
        all_references.patient_id.isin(construction_patients)
        & all_references.descriptor_id.isin(universe)
    ]
    construction_reference_matrix = reference_matrix(
        construction_references, cohort=target, descriptor_ids=universe
    )
    direct_ids = universe[:DIRECT_DESCRIPTOR_COUNT]
    direct_reference = reference_matrix(
        construction_references.loc[
            construction_references.descriptor_id.isin(direct_ids)
        ],
        cohort=target,
        descriptor_ids=direct_ids,
    )
    direct_draws, direct_flags = [
        frame.loc[frame.patient_id.isin(construction_patients)]
        for frame in (all_draws, all_flags)
    ]
    unit_matrices = {
        "target_reference": construction_reference_matrix,
        "source": source_pair_matrix(
            source_reference.loc[source_reference.descriptor_id.isin(direct_ids)],
            source_alternate.loc[source_alternate.descriptor_id.isin(direct_ids)],
            cohort=source,
            descriptor_ids=direct_ids,
        ),
        "synthetic": synthetic_matrix(
            direct_reference, direct_draws, direct_flags, cohort=target
        ),
        "observed": source_pair_matrix(
            references.loc[references.descriptor_id.isin(direct_ids)],
            alternates.loc[alternates.descriptor_id.isin(direct_ids)],
            cohort=target,
            descriptor_ids=direct_ids,
        ),
    }
    unit_audit = audit_unit_conversions(
        cfg,
        direction=direction,
        scope=scope,
        source_moments=source_joint,
        target_reference={
            descriptor: ReferenceMoments(
                moment.reference_mean, moment.reference_variance
            )
            for descriptor, moment in locked_predictions.items()
        },
        synthetic_moments=construction_synthetic,
        observed_moments=_joint_mapping(tables["matched_observed_moments"], universe),
        diagnostic_synthetic_moments=_joint_mapping(
            tables["matched_synthetic_moments"], universe
        ),
        source_patient_counts={
            descriptor: int(model.independent_patients)
            for descriptor, model in source_models.items()
        },
        matrices=unit_matrices,
    )
    unit_audit["source"], unit_audit["target"] = source, target
    tables["unit_conversion_control"] = unit_audit
    unit_failures = unit_audit.loc[unit_audit.passed.eq(False)]
    if not unit_failures.empty:
        failures = pd.DataFrame(
            [
                {
                    "descriptor_id": row.get("descriptor_id"),
                    "failure_stage": "unit_conversion_control",
                    "reason": "invariance_failed:"
                    + row["audit_component"]
                    + ":"
                    + row.get("input_role", ""),
                    "source": source,
                    "target": target,
                    "scope": scope,
                    "universe_count": len(universe),
                    "full_construction_universe_count": len(universe),
                    "patient_count": len(allowed_patients),
                    "conditioning": FULL_COHORT_CONDITIONING
                    if scope == "full"
                    else AUDIT_CONDITIONING,
                }
                for row in unit_failures.to_dict("records")
            ]
        )
        tables["evaluation_failures"] = pd.concat(
            [tables["evaluation_failures"], failures], ignore_index=True
        )
    paths.append(workspace / "audit/UNIT_CONTROL_SPECIFICATION.md")
    writer = RunWriter(
        workspace,
        cfg,
        "evaluation",
        run_id,
        inputs=list(dict.fromkeys(paths)),
        estimand="observed_single_contour_CCC_and_fixed_budget_selected_set_mean_target_CCC",
        role=f"post_commitment_target_evaluation:{direction}:{scope}",
        seeds={"stream": "empirical_bootstrap", "keys": [source, target, scope]},
    )
    for name, table in tables.items():
        writer.write_table(name + ".csv", table)
    return writer.finish(
        {
            "source": source,
            "target": target,
            "scope": scope,
            "construction_scope": construction_scope,
            "universe_sha256": content_hash(list(universe)),
            "universe_count": len(universe),
            "evaluation_patients": len(allowed_patients),
            "discovery_patients": int(target_patients.assignment.eq("discovery").sum()),
            "audit_patients": int(target_patients.assignment.eq("audit").sum()),
            "all_construction_commitments_verified": len(commitments),
            "coverage_nominal_allowed": bool(coverage_nominal_allowed),
            "unit_conversion_control_failed_checks": len(unit_failures),
            "technical_failure": not unit_failures.empty,
            "construction_scores_replaced_by_diagnostic_scores": False,
        }
    )
