"""Match error-scale models and evaluate ranking diagnostics."""

from __future__ import annotations
from redesign2.execution import read_text
from pathlib import Path
from redesign2.execution import copy_input
import json, math
from collections import Counter
import numpy as np
import pandas as pd
from redesign2.provenance import RunWriter, read_protocol, rng, subseed, content_hash
from redesign2.mechanism import (
    match_law,
    association,
    distribution,
    permutation_indices,
    permuted_associations,
    coverage_prediction,
    top_k,
)
from redesign2.simulation import (
    population_moments,
    population_transport,
    changed_real_law,
)
from redesign2.construction import source_pair_matrix, patient_cache_from_frame
from redesign2.empirical_math import aggregate_patient_moments
from redesign2.uncertainty import patient_bootstrap_multiplicities
from redesign2.transport import SourceModel, predict_target
from redesign2.moments import ReferenceMoments
from redesign2.execution import read_csv, storage_run_id, verify_frozen_input


def run(workspace, run_id):
    """Match error-scale models and evaluate ranking diagnostics."""
    workspace = Path(workspace).resolve()
    diagnostics = workspace / "critical_revision"
    analysis_spec = read_protocol(
        workspace / "config/measurement_validation_amendment_v2.yaml"
    )
    contour_spec = read_protocol(workspace / "config/protocol.lock.yaml")
    contour_runs = (
        workspace / "runs" / analysis_spec["revision"]["inherited_protocol_sha256"]
    )
    contour_summary = contour_runs / "study-report-unitprecision-v2"
    cell_columns = ["source", "target", "scope"]
    inputs = [
        Path(__file__),
        workspace / "src/redesign2/mechanism.py",
        workspace / "config/measurement_validation_amendment_v2.yaml",
        diagnostics / "AMENDMENT_LOCK.json",
        diagnostics / "descriptor_definition_audit.csv",
    ]
    lock = json.loads(read_text(diagnostics / "AMENDMENT_LOCK.json"))
    for name, sha in lock["input_hashes"].items():
        verify_frozen_input(workspace / name, sha)

    def read(path, **kw):
        path = Path(path)
        inputs.append(path)
        return read_csv(path, **kw)

    tables = {
        n: read(contour_summary / (n + ".csv"))
        for n in [
            "reference_evaluation_moments",
            "matched_observed_moments",
            "synthetic_evaluation_moments",
            "descriptor_prediction_errors",
            "descriptor_disagreement",
            "selected_set_utility",
            "primary_and_contextual_contrasts",
            "mechanism_coefficient_fields",
            "aggregation_audit",
            "profile_agreement",
            "bootstrap_replicates",
        ]
    }
    audit = (
        read(diagnostics / "descriptor_definition_audit.csv")
        .drop_duplicates("descriptor_id")
        .set_index("descriptor_id")
    )
    previous = (
        workspace
        / json.loads(read_text(workspace / "media_revision/analysis_package.json"))[
            "run"
        ]
    )
    coverage = read(previous / "coverage_diagnosis.csv")
    out = {}
    seeds = {}
    calibration = []
    parameters = []
    pop = []
    nullrows = []
    blockrows = []
    nullsummary = []
    nullindices = {}
    sens = []
    case = []
    pool = []
    splitrows = []
    splitids = []
    failrows = []
    faildetails = []
    trace = []
    recovery = []
    split = read(
        workspace / contour_spec["resolved_metadata"]["patient_split"], dtype=str
    )

    def subset(f, key):
        for c, v in zip(cell_columns, key, strict=True):
            f = f[f[c].eq(v)]
        return f

    def matrix_cache(target, scope, feature_ids):
        ref = read(
            contour_runs / f"prepared-20260915/{target}/reference_features.csv",
            dtype={"patient_id": str, "unit_id": str, "annotation_id": str},
        )
        alt = read(
            contour_runs / f"prepared-20260915/{target}/alternate_features.csv",
            dtype={"patient_id": str, "unit_id": str, "annotation_id": str},
        )
        if scope == "audit":
            patients = set(
                split.loc[
                    split.cohort.eq(target) & split.assignment.eq("audit"), "patient_id"
                ]
            )
            ref = ref[ref.patient_id.isin(patients)]
            alt = alt[alt.patient_id.isin(patients)]
        ref = ref[ref.descriptor_id.isin(feature_ids)]
        alt = alt[alt.descriptor_id.isin(feature_ids)]
        return source_pair_matrix(
            ref, alt, cohort=target, descriptor_ids=feature_ids
        ).patient_moments()

    def construction(key):
        source, target, scope = key
        cs = "full" if scope == "full" else "discovery"
        found = [
            p
            for n in [100, 200]
            if (
                p := (contour_runs / f"construct-{source}-to-{target}-{cs}-{n}")
            ).exists()
        ]
        return found[-1]

    def profile(cache, counts=None, original=None, minimum=10, coverage=0.9):
        return aggregate_patient_moments(
            cache,
            counts,
            min_independent_patients=minimum,
            min_weighted_support=coverage,
            locked_support=None if original is None else original.support,
        )

    # Match each direction and patient analysis separately.
    for key, reference_moments in tables["reference_evaluation_moments"].groupby(
        cell_columns, sort=True
    ):
        print("Cell", key, flush=True)
        identity = dict(zip(cell_columns, key))
        source, target, scope = key
        feature_ids = tuple(reference_moments.descriptor_id)
        coefficient_fields = tables["mechanism_coefficient_fields"]
        coefficient_fields = (
            coefficient_fields[coefficient_fields.cohort_template.eq(target)]
            .set_index("descriptor_id")
            .loc[list(feature_ids)]
            .reset_index()
        )
        observed_moments = (
            subset(tables["matched_observed_moments"], key)
            .set_index("descriptor_id")
            .loc[list(feature_ids)]
        )
        generated_moments = (
            subset(tables["synthetic_evaluation_moments"], key)
            .set_index("descriptor_id")
            .loc[list(feature_ids)]
        )
        for battery_name, positions in [
            ("complete", np.arange(len(feature_ids))),
            (
                "natural_zero",
                np.flatnonzero(
                    audit.loc[list(feature_ids)].natural_zero.to_numpy(bool)
                ),
            ),
        ]:
            b = coefficient_fields.iloc[positions].reset_index(drop=True)
            r = reference_moments.iloc[positions]
            for law in analysis_spec["analysis"]["laws"]:
                for theta in analysis_spec["analysis"]["theta"]:
                    laws = {}
                    for side, mom in [
                        ("observed", observed_moments),
                        ("synthetic", generated_moments),
                    ]:
                        m = mom.iloc[positions]
                        got, rec = match_law(
                            b,
                            r.reference_mean,
                            r.reference_variance,
                            m.disagreement_mse,
                            law=law,
                            theta=theta,
                            side=side,
                        )
                        ident = dict(
                            identity,
                            battery=battery_name,
                            law=law,
                            theta=theta,
                            side=side,
                        )
                        calibration.append(dict(ident, **rec, descriptors=len(b)))
                        if got is not None:
                            laws[side] = got
                            if battery_name == "complete":
                                for j, d in enumerate(b.descriptor_id):
                                    parameters.append(
                                        dict(
                                            ident,
                                            descriptor_id=d,
                                            family=b.family.iloc[j],
                                            **{
                                                name: float(getattr(got, name)[j])
                                                for name in [
                                                    "source_mean",
                                                    "source_variance",
                                                    "bias",
                                                    "slope",
                                                    "residual_sd",
                                                ]
                                            },
                                            heteroscedastic_log_sd_slope=got.heteroscedastic_log_sd_slope,
                                        )
                                    )
                    if len(laws) == 2:
                        a, c = (
                            population_moments(laws["observed"]),
                            population_moments(laws["synthetic"]),
                        )
                        pop.append(
                            dict(
                                identity,
                                battery=battery_name,
                                law=law,
                                theta=theta,
                                profile_spearman=association(a["ccc"], c["ccc"]),
                                descriptors=len(b),
                                matched=True,
                            )
                        )
                        if battery_name == "complete":
                            source_mom = a
                            for mult in [0.5, 1.0, 2.0, 4.0]:
                                for shift in [0.0, 1.0, 2.0]:
                                    mu = laws["observed"].source_mean + shift * np.sqrt(
                                        laws["observed"].source_variance
                                    )
                                    v = mult * laws["observed"].source_variance
                                    truth = population_moments(laws["observed"], mu, v)
                                    pred = population_transport(source_mom, mu, v)
                                    recovery.append(
                                        dict(
                                            identity,
                                            law=law,
                                            theta=theta,
                                            scenario="unchanged_conditional_law",
                                            variance_multiplier=mult,
                                            mean_shift_sd=shift,
                                            mean_absolute_error=float(
                                                np.mean(
                                                    abs(
                                                        np.array([p.ccc for p in pred])
                                                        - truth["ccc"]
                                                    )
                                                )
                                            ),
                                        )
                                    )
                            if law == "biased_proportional":
                                mu = laws["observed"].source_mean + np.sqrt(
                                    laws["observed"].source_variance
                                )
                                v = 2 * laws["observed"].source_variance
                                truth = population_moments(
                                    changed_real_law(laws["observed"], b), mu, v
                                )
                                pred = population_transport(source_mom, mu, v)
                                recovery.append(
                                    dict(
                                        identity,
                                        law=law,
                                        theta=theta,
                                        scenario="changed_conditional_law",
                                        variance_multiplier=2.0,
                                        mean_shift_sd=1.0,
                                        mean_absolute_error=float(
                                            np.mean(
                                                abs(
                                                    np.array([p.ccc for p in pred])
                                                    - truth["ccc"]
                                                )
                                            )
                                        ),
                                    )
                                )
                    else:
                        pop.append(
                            dict(
                                identity,
                                battery=battery_name,
                                law=law,
                                theta=theta,
                                profile_spearman=np.nan,
                                descriptors=len(b),
                                matched=False,
                            )
                        )
        # Preserve feature-family and region blocks in the permutation comparison.
        benchmark = subset(tables["profile_agreement"], key)
        benchmark = float(
            benchmark[benchmark.policy.eq("source_ccc")].profile_spearman.iloc[0]
        )
        obs = association(
            observed_moments.ccc.to_numpy(), generated_moments.ccc.to_numpy()
        )
        for design in ["family_region_cluster", "unrestricted_sensitivity"]:
            generator = rng(analysis_spec, "permutation", *key, design)
            seeds["permutation:" + ":".join(key) + ":" + design] = subseed(
                analysis_spec, "permutation", *key, design
            )
            indices, blocks = permutation_indices(
                feature_ids,
                reference_moments.family.to_list(),
                audit.loc[list(feature_ids)].region.to_list(),
                repetitions=analysis_spec["analysis"]["permutations"],
                rng=generator,
                unrestricted=design.startswith("unrestricted"),
            )
            null = permuted_associations(
                observed_moments.ccc.to_numpy(),
                generated_moments.reference_variance.to_numpy(),
                generated_moments.reference_disagreement_covariance.to_numpy(),
                generated_moments.disagreement_mse.to_numpy(),
                indices,
            )
            nullindices["__".join((*key, design))] = indices.astype(np.int16)
            total = math.prod((int(b["distinct_permutations"]) for b in blocks))
            for b in blocks:
                blockrows.append(
                    dict(
                        identity,
                        design=design,
                        **b,
                        total_distinct_permutations=str(total),
                    )
                )
            for n, x in enumerate(null):
                nullrows.append(
                    dict(identity, design=design, replicate=n, null_profile_spearman=x)
                )
            defined = null[np.isfinite(null)]
            percentile = (
                float(
                    (np.sum(defined < obs) + 0.5 * np.sum(defined == obs))
                    / len(defined)
                )
                if len(defined)
                else np.nan
            )
            nullsummary.append(
                dict(
                    identity,
                    design=design,
                    **distribution(null),
                    observed_profile_spearman=obs,
                    observed_percentile=percentile,
                    source_real_benchmark=benchmark,
                    benchmark_not_latent_coupling_estimate=True,
                    total_distinct_permutations=str(total),
                )
            )
        # Supplement S6: retained feature sets and measurement-scale diagnostics.
        pred = subset(tables["descriptor_prediction_errors"], key)
        keep = tuple(
            (d for d in feature_ids if not bool(audit.loc[d, "denominator_risk"]))
        )
        for battery_name, battery_ids in [
            ("complete", feature_ids),
            ("denominator_defined", keep),
        ]:
            scores = {
                p: pred[pred.policy.eq(p)]
                .set_index("descriptor_id")
                .loc[list(battery_ids)]
                .construction_score.to_numpy()
                for p in ["source_ccc", "synthetic_ccc", "error_transport"]
            }
            target_ccc = (
                pred[pred.policy.eq("source_ccc")]
                .set_index("descriptor_id")
                .loc[list(battery_ids)]
                .observed_target_ccc.to_numpy()
            )
            ordered = top_k(battery_ids, target_ccc)
            ur = float(np.mean(target_ccc))
            uo = float(np.mean(target_ccc[list(ordered)])) if ordered else np.nan
            utilities = {}
            dg = (
                subset(tables["descriptor_disagreement"], key)
                .set_index("descriptor_id")
                .loc[list(battery_ids)]
            )
            for policy, s in scores.items():
                selected = top_k(battery_ids, s)
                u = float(np.mean(target_ccc[list(selected)])) if selected else np.nan
                utilities[policy] = u
                sens.append(
                    dict(
                        identity,
                        battery=battery_name,
                        policy=policy,
                        J=len(battery_ids),
                        k=max(1, len(battery_ids) // 4),
                        profile_spearman=association(s, target_ccc),
                        ccc_mae=float(np.mean(abs(s - target_ccc))),
                        U=u,
                        U_random=ur,
                        U_oracle=uo,
                        G=(u - ur) / (uo - ur) if uo - ur > 1e-12 else np.nan,
                        selected_ids=json.dumps([battery_ids[j] for j in selected])
                        if selected
                        else "[]",
                        median_absolute_log2_rmse_ratio=float(
                            dg.log2_rmse_ratio.abs().median()
                        ),
                        median_absolute_bias_difference=float(
                            dg.standardized_bias_difference.abs().median()
                        ),
                        median_absolute_slope_difference=float(
                            dg.reference_slope_difference.abs().median()
                        ),
                    )
                )
            for row in sens[-3:]:
                row["synthetic_minus_source"] = (
                    utilities["synthetic_ccc"] - utilities["source_ccc"]
                )
        cr = construction(key)
        models = (
            read(cr / "source_models.csv")
            .set_index("descriptor_id")
            .loc[list(feature_ids)]
        )
        for j, d in enumerate(feature_ids):
            ratio = float(
                np.log2(
                    reference_moments.reference_variance.iloc[j]
                    / models.source_reference_variance.iloc[j]
                )
            )
            error = (
                pred[pred.policy.eq("error_transport")]
                .set_index("descriptor_id")
                .loc[d, "absolute_ccc_error"]
            )
            case.append(
                dict(
                    identity,
                    descriptor_id=d,
                    family=reference_moments.family.iloc[j],
                    region=audit.loc[d, "region"],
                    log2_reference_variance_ratio=ratio,
                    transport_absolute_error=error,
                )
            )
        conv = (
            read(cr / "draw_convergence.csv")
            .set_index("descriptor_id")
            .loc[list(feature_ids)]
        )
        summary = read(cr / "draw_convergence_summary.csv").iloc[0]
        truth = (
            pred[pred.policy.eq("source_ccc")]
            .set_index("descriptor_id")
            .loc[list(feature_ids)]
            .observed_target_ccc.to_numpy()
        )
        D = float(
            subset(tables["selected_set_utility"], key)
            .query("policy=='source_ccc'")
            .utility.iloc[0]
        )
        earlier = top_k(feature_ids, conv.earlier_synthetic_ccc.to_numpy())
        final = top_k(feature_ids, conv.final_synthetic_ccc.to_numpy())
        u1 = float(np.mean(truth[list(earlier)])) if earlier else np.nan
        u2 = float(np.mean(truth[list(final)])) if final else np.nan
        pool.append(
            dict(
                identity,
                **summary.to_dict(),
                earlier_selected_ids=json.dumps([feature_ids[j] for j in earlier])
                if earlier
                else "[]",
                final_selected_ids=json.dumps([feature_ids[j] for j in final])
                if final
                else "[]",
                top_k_overlap_fraction=len(set(earlier) & set(final)) / len(final)
                if earlier and final
                else np.nan,
                U_earlier=u1,
                U_final=u2,
                utility_change=u2 - u1,
                source_U=D,
                synthetic_minus_source_earlier=u1 - D,
                synthetic_minus_source_final=u2 - D,
                headline_sign_changes=bool(np.sign(u1 - D) != np.sign(u2 - D))
                if np.isfinite(u1 + u2)
                else None,
            )
        )
        # Supplement S8: patient split halves and complete bootstrap guard records.
        observed_cache = matrix_cache(target, scope, feature_ids)
        observed_original = profile(observed_cache)
        for repetition in range(analysis_spec["analysis"]["split_half_repetitions"]):
            gen = rng(analysis_spec, "split_half", target, scope, repetition)
            order = gen.permutation(len(observed_cache.patient_ids))
            cut = len(order) // 2
            a = np.zeros(len(order), int)
            a[order[:cut]] = 1
            b = 1 - a
            pa, pb = (profile(observed_cache, a), profile(observed_cache, b))
            x = np.array([m.ccc for m in pa.moments])
            y = np.array([m.ccc for m in pb.moments])
            sx, sy = (top_k(feature_ids, x), top_k(feature_ids, y))
            finite = np.isfinite(x) & np.isfinite(y)
            splitrows.append(
                dict(
                    identity,
                    replicate=repetition,
                    patient_count=len(order),
                    left_patients=cut,
                    right_patients=len(order) - cut,
                    descriptors=len(feature_ids),
                    common_evaluable=int(finite.sum()),
                    split_half_spearman=association(x, y),
                    top_k_overlap=len(set(sx) & set(sy)) / len(sx)
                    if sx and sy
                    else np.nan,
                    reason=""
                    if finite.sum() >= 3
                    else "insufficient_eligible_half_patient_support",
                )
            )
            splitids.append(
                dict(
                    identity,
                    replicate=repetition,
                    left=json.dumps(
                        [observed_cache.patient_ids[j] for j in order[:cut]]
                    ),
                    right=json.dumps(
                        [observed_cache.patient_ids[j] for j in order[cut:]]
                    ),
                )
            )
        reference_cache = (
            patient_cache_from_frame(
                read(
                    cr / "target_reference_patient_moments.csv",
                    dtype={"patient_id": str},
                ),
                descriptor_ids=feature_ids,
            )
            if scope == "full"
            else None
        )
        synthetic_cache = (
            patient_cache_from_frame(
                read(cr / "synthetic_patient_moments.csv", dtype={"patient_id": str}),
                descriptor_ids=feature_ids,
            )
            if scope == "full"
            else None
        )
        rp = (
            profile(
                reference_cache,
                minimum=1,
                coverage=contour_spec["features"][
                    "min_construction_weighted_availability"
                ],
            )
            if reference_cache
            else None
        )
        sp = (
            profile(
                synthetic_cache,
                minimum=1,
                coverage=contour_spec["features"][
                    "min_construction_weighted_availability"
                ],
            )
            if synthetic_cache
            else None
        )
        modelmaps = {}
        for d, row in models.iterrows():
            vals = {name: row[name] for name in SourceModel.__dataclass_fields__}
            vals["reason"] = None if pd.isna(vals["reason"]) else vals["reason"]
            modelmaps[d] = SourceModel(**vals)
        original_scores = {
            p: pred[pred.policy.eq(p)]
            .set_index("descriptor_id")
            .loc[list(feature_ids)]
            .construction_score.to_numpy()
            for p in ["source_ccc", "synthetic_ccc", "error_transport"]
        }
        draws = patient_bootstrap_multiplicities(
            len(observed_cache.patient_ids),
            analysis_spec["analysis"]["bootstrap_replicates"],
            rng=rng(contour_spec, "empirical_bootstrap", *key),
        )
        original_stored = subset(tables["bootstrap_replicates"], key).sort_values(
            "replicate"
        )
        counts_summary = Counter()
        descriptor_summary = Counter()
        for n, counts in enumerate(draws):
            op = profile(observed_cache, counts, observed_original)
            observed = np.array([m.ccc for m in op.moments])
            maps = {p: s.copy() for p, s in original_scores.items()}
            if scope == "full":
                refp = profile(
                    reference_cache,
                    counts,
                    rp,
                    minimum=1,
                    coverage=contour_spec["features"][
                        "min_construction_weighted_availability"
                    ],
                )
                synp = profile(
                    synthetic_cache,
                    counts,
                    sp,
                    minimum=1,
                    coverage=contour_spec["features"][
                        "min_construction_weighted_availability"
                    ],
                )
                maps["synthetic_ccc"] = np.array([m.ccc for m in synp.moments])
                maps["error_transport"] = np.array(
                    [
                        predict_target(
                            modelmaps[d],
                            ReferenceMoments(m.reference_mean, m.reference_variance),
                        ).ccc
                        if s.evaluable
                        else np.nan
                        for d, m, s in zip(
                            feature_ids, refp.moments, refp.support, strict=True
                        )
                    ]
                )
            for battery_name, positions in [
                ("complete", np.arange(len(feature_ids))),
                (
                    "denominator_defined",
                    np.array([j for j, d in enumerate(feature_ids) if d in keep]),
                ),
            ]:
                bid = tuple((feature_ids[j] for j in positions))
                bscores = {p: s[positions] for p, s in maps.items()}
                target_values = observed[positions]
                ut = {}
                for policy, s in bscores.items():
                    ranking = (
                        s if scope == "full" else original_scores[policy][positions]
                    )
                    selected = top_k(bid, ranking)
                    reason = ""
                    if selected is None:
                        reason = (
                            "undefined_synthetic_ranking_score"
                            if policy == "synthetic_ccc"
                            else "undefined_source_ranking_score"
                            if policy == "source_ccc"
                            else "other_numerical_failure"
                        )
                    elif len(selected) != max(1, len(bid) // 4):
                        reason = "incomplete_top_k"
                    elif not np.isfinite(target_values[list(selected)]).all():
                        reason = "undefined_selected_target_utility"
                    u = (
                        float(np.mean(target_values[list(selected)]))
                        if selected and (not reason)
                        else np.nan
                    )
                    ut[policy] = u
                    counts_summary[battery_name, policy, reason or "defined"] += 1
                    if reason:
                        for j in positions:
                            bad_rank = (
                                not np.isfinite(maps[policy][j])
                                if scope == "full"
                                else not np.isfinite(original_scores[policy][j])
                            )
                            bad_criterion = (
                                selected is not None
                                and list(positions).index(j) in selected
                                and (not np.isfinite(observed[j]))
                            )
                            if bad_rank or bad_criterion:
                                detail = (
                                    op.support[j].reason
                                    if bad_criterion
                                    else synp.support[j].reason
                                    if policy == "synthetic_ccc" and scope == "full"
                                    else "undefined_ranking_moment"
                                )
                                code = (
                                    "insufficient_patient_support"
                                    if detail
                                    and ("support" in detail or "patient" in detail)
                                    else "invalid_CCC_denominator"
                                    if detail
                                    and (
                                        "denominator" in detail or "constant" in detail
                                    )
                                    else "undefined_descriptor_value"
                                    if detail
                                    and ("defined" in detail or "empty" in detail)
                                    else "other_numerical_failure"
                                )
                                descriptor_summary[
                                    battery_name,
                                    policy,
                                    feature_ids[j],
                                    reason,
                                    code,
                                    str(detail),
                                ] += 1
                        faildetails.append(
                            dict(
                                identity,
                                replicate=n,
                                battery=battery_name,
                                policy=policy,
                                reason=reason,
                            )
                        )
                    if battery_name == "complete":
                        expected = original_stored.iloc[n]["utility:" + policy]
                        assert np.isclose(
                            u, expected, equal_nan=True, rtol=1e-10, atol=1e-10
                        ), (key, n, policy, u, expected)
                trace.append(
                    dict(
                        identity,
                        battery=battery_name,
                        replicate=n,
                        **{"utility:" + p: v for p, v in ut.items()},
                        synthetic_minus_source=ut["synthetic_ccc"] - ut["source_ccc"],
                    )
                )
        for (battery_name, policy, reason), count in counts_summary.items():
            failrows.append(
                dict(
                    identity,
                    battery=battery_name,
                    policy=policy,
                    reason=reason,
                    replicates=count,
                )
            )
        for (
            battery_name,
            policy,
            d,
            reason,
            code,
            detail,
        ), count in descriptor_summary.items():
            failrows.append(
                dict(
                    identity,
                    battery=battery_name,
                    policy=policy,
                    reason=reason,
                    descriptor_id=d,
                    detail_code=code,
                    detail=detail,
                    denominator_risk=bool(audit.loc[d, "denominator_risk"]),
                    replicates=count,
                )
            )
    coverage = coverage.copy().rename(
        columns={"coverage_unconditional": "coverage_all_datasets"}
    )
    coverage["predicted_coverage"] = coverage_prediction(coverage.absolute_bias_over_sd)
    coverage["predicted_minus_observed"] = (
        coverage.predicted_coverage - coverage.coverage_all_datasets
    )
    coverage["endpoint_type"] = np.where(
        coverage.endpoint.str.startswith("utility:"), "utility", "contrast"
    )
    coverage_summary = pd.DataFrame(
        [
            dict(
                endpoints=len(coverage),
                prediction_correlation=float(
                    coverage[["predicted_coverage", "coverage_all_datasets"]]
                    .corr()
                    .iloc[0, 1]
                ),
                mean_absolute_prediction_error=float(
                    coverage.predicted_minus_observed.abs().mean()
                ),
                minimum_residual=float(coverage.predicted_minus_observed.min()),
                maximum_residual=float(coverage.predicted_minus_observed.max()),
                locked_bias_gate_changed=False,
            )
        ]
    )
    aggregation = tables["aggregation_audit"]
    aggregation = aggregation[aggregation.origin.eq("reference_only_empirical")].copy()
    ag = []
    for (cohort, k), group in aggregation.groupby(["cohort", "K"]):
        p = group.pivot(index="descriptor_id", columns="estimator", values="ccc")
        if "ccc_ensemble_mean_K" not in p or "ccc_single_measurement_moments" not in p:
            continue
        difference = p.ccc_ensemble_mean_K - p.ccc_single_measurement_moments
        for d in p.index:
            ag.append(
                dict(
                    cohort=cohort,
                    K=k,
                    descriptor_id=d,
                    single_contour=p.loc[d, "ccc_single_measurement_moments"],
                    draw_averaged=p.loc[d, "ccc_ensemble_mean_K"],
                    increase=difference.loc[d],
                )
            )
    out = {
        "matching_calibration": calibration,
        "matched_law_parameters": parameters,
        "matched_population_association": pop,
        "permutation_replicates": nullrows,
        "permutation_blocks": blockrows,
        "permutation_summary": nullsummary,
        "denominator_sensitivity": sens,
        "case_mix_variance_shift": case,
        "finite_pool_selection_stability": pool,
        "split_half_replicates": splitrows,
        "split_half_patient_ids": splitids,
        "undefined_reason_counts": failrows,
        "undefined_replicate_reasons": faildetails,
        "bootstrap_trace": trace,
        "matched_transport_recovery": recovery,
        "aggregation_effects": ag,
    }
    frames = {name: pd.DataFrame(rows) for name, rows in out.items()}
    frames.update(
        coverage_prediction=coverage, coverage_prediction_summary=coverage_summary
    )
    for name, column, keys in [
        ("split_half_replicates", "split_half_spearman", cell_columns),
        ("aggregation_effects", "increase", ["cohort", "K"]),
        ("case_mix_variance_shift", "log2_reference_variance_ratio", cell_columns),
    ]:
        frames[name.replace("_replicates", "") + "_summary"] = pd.DataFrame(
            [
                dict(
                    zip(keys, key if isinstance(key, tuple) else (key,)),
                    **distribution(g[column]),
                )
                for key, g in frames[name].groupby(keys, sort=True)
            ]
        )
    writer = RunWriter(
        workspace,
        analysis_spec,
        "critical_revision_existing_data",
        storage_run_id(run_id, "existing-data-v1"),
        inputs=list(dict.fromkeys(inputs)),
        role="post_analysis_descriptive_no_primary_replacement",
        estimand="unchanged_single_random_contour_and_definition_sensitivities",
        seeds=seeds,
    )
    for name, f in frames.items():
        writer.write_table(name + ".csv", f)
    np.savez_compressed(writer.temp / "permutation_indices.npz", **nullindices)
    copy_input(__file__, writer.temp / "analysis_source.py")
    copy_input(
        workspace / "src/redesign2/mechanism.py",
        writer.temp / "critical_revision_source.py",
    )
    writer.write_json(
        "reason_vocabulary.json",
        {
            "codes": [
                "undefined_descriptor_value",
                "invalid_CCC_denominator",
                "insufficient_patient_support",
                "undefined_source_ranking_score",
                "undefined_synthetic_ranking_score",
                "incomplete_top_k",
                "undefined_selected_target_utility",
                "zero_random_oracle_denominator",
                "other_numerical_failure",
            ],
            "zero_count_codes_are_retained": True,
        },
    )
    final = writer.finish()
    (diagnostics / "analysis_package.json").write_text(
        json.dumps(
            {
                "run": str(final.relative_to(workspace)).replace("\\", "/"),
                "configuration_sha256": content_hash(analysis_spec),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Completed", final)
