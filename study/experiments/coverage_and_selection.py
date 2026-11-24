"""Compute fixed-set selection anchors and diagnose saved interval coverage."""

from __future__ import annotations
from redesign2.execution import read_text
from pathlib import Path
from redesign2.execution import copy_input
import json
import numpy as np
import pandas as pd
from redesign2.provenance import (
    RunWriter,
    file_hash,
    content_hash,
    read_protocol,
    rng,
    subseed,
    verify_run,
)
from redesign2.execution import read_csv, storage_run_id, verify_frozen_input


def run(workspace, run_id):
    """Compute fixed-set selection anchors and diagnose saved interval coverage."""
    WORK = Path(workspace).resolve()
    OLD = WORK / "runs/8139ac7fc83dcba4a345f3b76636bacede3497c3d503d19e32d05d9083504a8d"
    REPORT = OLD / "study-report-unitprecision-v2"
    REV = WORK / "media_revision"
    FILES = [
        "reference_evaluation_moments",
        "mechanism_coefficient_fields",
        "descriptor_prediction_errors",
        "selected_set_utility",
        "primary_and_contextual_contrasts",
        "descriptor_disagreement",
        "matched_observed_moments",
        "simulation_finite_sample_mechanism",
        "profile_agreement",
        "coverage_fixed_sets",
        "coverage_summary",
    ]
    INPUTS = [REPORT / (n + ".csv") for n in FILES] + [
        OLD / "coverage-final/coverage_repetitions.csv"
    ]
    KEYS = ["source", "target", "scope"]

    def anchors(prediction, selections, contrasts, cfg):
        rows = []
        for key, group in selections.groupby(KEYS, sort=True):
            data = prediction
            for k, v in zip(KEYS, key, strict=True):
                data = data[data[k].eq(v)]
            p = data[data.policy.eq("source_ccc")].sort_values("descriptor_id")
            assert not p.descriptor_id.duplicated().any()
            k = int(group.budget.iloc[0])
            J = int(group.universe_count.iloc[0])
            assert len(p) == J
            target = p.set_index("descriptor_id").observed_target_ccc
            defined = np.isfinite(target).all()
            random = float(target.mean()) if defined else np.nan
            ordered = (
                sorted(target.index, key=lambda d: (-target[d], d)) if defined else []
            )
            oracle = float(target.loc[ordered[:k]].mean()) if defined else np.nan
            gap = oracle - random
            generator = rng(cfg, "anchor_check", *key)
            estimates = (
                np.array(
                    [
                        float(target.iloc[generator.choice(J, k, replace=False)].mean())
                        for _ in range(cfg["analysis"]["uniform_subsets"])
                    ]
                )
                if defined
                else np.array([])
            )
            mc = float(estimates.mean()) if defined else np.nan
            mcse = (
                float(estimates.std(ddof=1) / np.sqrt(len(estimates)))
                if defined
                else np.nan
            )
            assert not defined or abs(mc - random) <= max(6 * mcse, 1e-12), (
                key,
                mc,
                random,
                mcse,
            )
            cd = contrasts
            for c, v in zip(KEYS, key, strict=True):
                cd = cd[cd[c].eq(v)]
            direct = cd[
                cd["first"].eq("synthetic_ccc") & cd["second"].eq("source_ccc")
            ].iloc[0]
            for _, r in group.iterrows():
                selected = json.loads(r.selected_ids)
                assert (
                    len(selected) == k
                    and len(set(selected)) == k
                    and (set(selected) <= set(target.index))
                )
                utility = (
                    float(target.loc[selected].mean())
                    if np.isfinite(target.loc[selected]).all()
                    else np.nan
                )
                assert np.isclose(utility, r.utility, atol=1e-12, equal_nan=True)
                count = len(set(selected) & set(ordered[:k])) if defined else np.nan
                gain = (
                    (utility - random) / gap
                    if defined
                    and gap > cfg["analysis"]["anchor_denominator_min"]
                    and np.isfinite(utility)
                    else np.nan
                )
                rows.append(
                    dict(
                        zip(KEYS, key),
                        policy=r.policy,
                        J=J,
                        k=k,
                        U_random=random,
                        U_oracle=oracle,
                        U=utility,
                        G=gain,
                        anchor_reason=""
                        if defined
                        else "undefined_CCC_in_complete_eligible_battery",
                        gain_reason=""
                        if np.isfinite(gain)
                        else "undefined_or_numerically_unstable_random_to_oracle_gap",
                        random_to_oracle_gap=gap,
                        random_mc=mc,
                        random_mc_standard_error=mcse,
                        random_mc_draws=len(estimates),
                        precision_count=count,
                        precision_fraction=count / k,
                        chance_precision=k / J,
                        chance_count=k * k / J,
                        selected_ids=r.selected_ids,
                        oracle_ids=json.dumps(ordered[:k]),
                        utility_lower=r.interval_lower,
                        utility_upper=r.interval_upper,
                        utility_interval_kind=r.interval_kind,
                        synthetic_minus_source=direct.difference,
                        direct_lower=direct.interval_lower,
                        direct_upper=direct.interval_upper,
                        direct_interval_kind=direct.interval_kind,
                        direct_undefined_fraction=direct.undefined_fraction,
                    )
                )
        return pd.DataFrame(rows)

    def coverage_diagnosis(path, cfg):
        frame = read_csv(path)
        selected = frame[frame.endpoint.str.startswith(("utility:", "contrast:"))]
        rows = []
        for key, g in selected.groupby(["template", "regime", "endpoint"], sort=True):
            truth = g.population_truth.unique()
            assert len(truth) == 1
            finite = np.isfinite(g.point_estimate)
            issued = g[g.interval_issued]
            estimated = g.loc[finite, "point_estimate"]
            bias = float(estimated.mean() - truth[0])
            sd = float(estimated.std(ddof=1))
            middle = (issued.percentile_lower + issued.percentile_upper) / 2
            rows.append(
                dict(
                    template=key[0],
                    law=key[1],
                    endpoint=key[2],
                    patient_count=int(g.patient_count.iloc[0]),
                    datasets=len(g),
                    finite_estimates=len(estimated),
                    population_truth=truth[0],
                    mean_estimate=float(estimated.mean()),
                    signed_bias=bias,
                    absolute_bias=abs(bias),
                    empirical_sd=sd,
                    absolute_bias_over_sd=abs(bias) / sd if sd > 0 else np.nan,
                    mean_absolute_estimation_error=float(
                        abs(estimated - truth[0]).mean()
                    ),
                    mean_nominal_width=float(
                        (issued.percentile_upper - issued.percentile_lower).mean()
                    ),
                    coverage_issued=float(issued.covered.mean()),
                    coverage_unconditional=float(g.covered.mean()),
                    issued_count=len(issued),
                    issued_fraction=len(issued) / len(g),
                    signed_midpoint_bias=float(middle.mean() - truth[0]),
                    central_estimate_p025=float(estimated.quantile(0.025)),
                    central_estimate_p975=float(estimated.quantile(0.975)),
                )
            )
        table = pd.DataFrame(rows)
        cells = []
        for key, g in table[table.endpoint.str.startswith("utility:")].groupby(
            ["template", "law"]
        ):
            qualifies = (
                g.absolute_bias_over_sd.ge(cfg["analysis"]["gate_min_utility_bias_sd"])
                & g.coverage_unconditional.lt(cfg["analysis"]["gate_coverage_max"])
                & (np.sign(g.signed_bias) == np.sign(g.signed_midpoint_bias))
            )
            cells.append(
                dict(
                    template=key[0],
                    law=key[1],
                    utility_endpoints=len(g),
                    qualifying_endpoints=int(qualifies.sum()),
                    gate_passed=bool(
                        qualifies.sum()
                        >= cfg["analysis"]["gate_min_qualifying_utilities_per_cell"]
                    ),
                )
            )
        gate = dict(
            rule="At least two of four utilities in every existing cell: bias/SD>=1, undercoverage and concordant midpoint bias",
            supported=all((c["gate_passed"] for c in cells)),
            cells=cells,
            inference="Evidence consistent with a finite-sample bias contribution"
            if all((c["gate_passed"] for c in cells))
            else "Cause unresolved; demote detailed coverage diagnosis to supplement",
            empirical_ranges_remain_sampling_stability=True,
            replacement_interval=False,
            sample_size_controlled_intervention=False,
        )
        return (table, gate)

    def degeneracy(disagreement, matched, cfg):
        rows = []
        central = []
        for key, g in disagreement.groupby(KEYS, sort=True):
            m = matched
            for k, v in zip(KEYS, key, strict=True):
                m = m[m[k].eq(v)]
            m = m.set_index("descriptor_id")
            for name in [
                "standardized_bias_difference",
                "reference_slope_difference",
                "log2_rmse_ratio",
            ]:
                vals = g[name].to_numpy()
                finite = vals[np.isfinite(vals)]
                lo, hi = np.quantile(
                    finite, cfg["analysis"]["central_display_quantiles"]
                )
                central.append(
                    dict(
                        zip(KEYS, key),
                        endpoint=name,
                        descriptors=len(g),
                        finite=len(finite),
                        median=float(np.median(finite)),
                        q05=lo,
                        q95=hi,
                        min=float(finite.min()),
                        max=float(finite.max()),
                        below_window=int((finite < lo).sum()),
                        above_window=int((finite > hi).sum()),
                    )
                )
                if name == "log2_rmse_ratio":
                    continue
                selected = (
                    g.assign(absolute=g[name].abs())
                    .sort_values(["absolute", "descriptor_id"], ascending=[False, True])
                    .head(cfg["analysis"]["diagnostic_extremes_per_endpoint"])
                )
                for rank, (_, r) in enumerate(selected.iterrows(), 1):
                    mm = m.loc[r.descriptor_id]
                    rows.append(
                        dict(
                            zip(KEYS, key),
                            endpoint=name,
                            rank=rank,
                            descriptor_id=r.descriptor_id,
                            family=r.family,
                            value=r[name],
                            reference_mean=mm.reference_mean,
                            reference_variance=mm.reference_variance,
                            reference_cv_squared=mm.reference_variance
                            / mm.reference_mean**2
                            if mm.reference_mean
                            else np.nan,
                            observed_covariance=mm.reference_disagreement_covariance,
                            observed_disagreement_mse=mm.disagreement_mse,
                            synthetic_slope=r.synthetic_reference_slope,
                            observed_slope=r.observed_reference_slope,
                            observed_rmse=float(np.sqrt(r.observed_disagreement_mse)),
                            classification="small_relative_reference_dispersion"
                            if mm.reference_mean
                            and mm.reference_variance / mm.reference_mean**2 < 1e-06
                            else "retained_extreme_no_causal_assignment",
                        )
                    )
        return (pd.DataFrame(rows), pd.DataFrame(central))

    cfg = read_protocol(WORK / "config/media_revision_20260917.yaml")
    lock = json.loads(read_text(WORK / "MEDIA_REVISION_LOCK.json", encoding="utf-8"))
    assert lock["configuration_sha256"] == content_hash(cfg)
    for name, sha in lock["locked_files"].items():
        verify_frozen_input(WORK / name, sha)
    data = {name: read_csv(REPORT / (name + ".csv")) for name in FILES}
    inputs = INPUTS + [
        WORK / "MEDIA_REVISION_AMENDMENT.md",
        WORK / "MEDIA_REVISION_LOCK.json",
        WORK / "config/media_revision_20260917.yaml",
        Path(__file__),
        WORK / "V16_SNAPSHOT.json",
    ]
    writer = RunWriter(
        WORK,
        cfg,
        "media_revision_reanalysis",
        storage_run_id(run_id, "existing-data-reanalysis-v1"),
        inputs=inputs,
        role="locked_post_v16_development_reanalysis_no_new_imaging_draws",
        estimand="unchanged_single_contour_CCC_with_descriptive_selection_anchors_and_saved_coverage_diagnosis",
        seeds={
            "anchor_check": {
                ":".join(key): str(subseed(cfg, "anchor_check", *key))
                for key in data["selected_set_utility"].groupby(KEYS).groups
            },
            "coefficient_fields": "exact_stored_v16_fields_no_new_coefficient_seeds",
            "coverage": "exact_stored_v16_repetitions",
        },
    )
    tables = {}
    tables["anchored_utility"] = anchors(
        data["descriptor_prediction_errors"],
        data["selected_set_utility"],
        data["primary_and_contextual_contrasts"],
        cfg,
    )
    tables["coverage_diagnosis"], gate = coverage_diagnosis(
        OLD / "coverage-final/coverage_repetitions.csv", cfg
    )
    tables["extreme_descriptor_diagnosis"], tables["central_disagreement_windows"] = (
        degeneracy(
            data["descriptor_disagreement"], data["matched_observed_moments"], cfg
        )
    )
    for name in [
        "profile_agreement",
        "descriptor_prediction_errors",
        "descriptor_disagreement",
        "primary_and_contextual_contrasts",
    ]:
        tables[name] = data[name].copy()
        if name == "primary_and_contextual_contrasts":
            tables[name]["original_primary"] = tables[name]["primary"]
            tables[name]["revision_primary"] = tables[name]["first"].eq(
                "synthetic_ccc"
            ) & tables[name]["second"].eq("source_ccc")
    for name, table in tables.items():
        writer.write_table(name + ".csv", table)
    writer.write_json("coverage_decision.json", gate)
    source = writer.temp / "sources"
    source.mkdir()
    for p in [
        Path(__file__),
        WORK / "config/media_revision_20260917.yaml",
        WORK / "MEDIA_REVISION_AMENDMENT.md",
        WORK / "MEDIA_REVISION_LOCK.json",
    ]:
        copy_input(p, source / p.name)
    final = writer.finish(
        {
            "optional_RQ5": "not_run",
            "unchanged_original_selected_sets": True,
            "original_configuration_unchanged": True,
        }
    )
    assert not verify_run(final)
    REV.mkdir(exist_ok=True)
    (REV / "analysis_package.json").write_text(
        json.dumps(
            {
                "run": str(final.relative_to(WORK)).replace("\\", "/"),
                "manifest_sha256": file_hash(final / "run_manifest.json"),
                "coverage_gate_supported": gate["supported"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Sealed revision reanalysis:", final)
    print("Coverage gate:", gate["inference"])
    print(
        tables["anchored_utility"][
            KEYS
            + ["policy", "U_random", "U_oracle", "U", "G", "synthetic_minus_source"]
        ].to_string(index=False)
    )
    print(
        tables["coverage_diagnosis"][
            [
                "template",
                "law",
                "endpoint",
                "signed_bias",
                "empirical_sd",
                "absolute_bias_over_sd",
                "coverage_unconditional",
            ]
        ].to_string(index=False)
    )
