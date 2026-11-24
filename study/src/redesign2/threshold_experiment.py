"""Run the review-driven CCC threshold experiment from immutable saved measurements."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .construction import patient_cache_from_frame, source_pair_matrix
from .empirical_math import aggregate_patient_moments
from .policies import construct_screens, evaluate_selection
from .provenance import (
    RunWriter,
    content_hash,
    file_hash,
    read_protocol,
    rng,
    resolve_input_path,
)
from .threshold_decisions import (
    ALL_METRICS,
    CONTRAST_METRICS,
    decision_contrasts,
    summarize_replicates,
    threshold_decision,
)
from .uncertainty import (
    AUDIT_CONDITIONING,
    FULL_COHORT_CONDITIONING,
    patient_bootstrap_multiplicities,
)


def _verified_output(
    root: Path, manifest: dict, name: str, inputs: list[Path]
) -> pd.DataFrame:
    path = root / name
    if manifest["outputs"].get(name) != file_hash(path):
        raise ValueError(f"Inherited output is missing or changed: {path}")
    inputs.append(path)
    return pd.read_csv(
        path,
        float_precision="round_trip",
        dtype={"patient_id": str, "descriptor_id": str},
        keep_default_na=True,
    )


def _input_path(manifest: dict, predicate, inputs: list[Path]) -> Path:
    matches = [
        resolve_input_path(path)
        for path, expected in manifest["inputs"].items()
        if predicate(Path(path))
        and resolve_input_path(path).is_file()
        and file_hash(path) == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one verified inherited input, found {len(matches)}")
    inputs.append(matches[0])
    return matches[0]


def _mapping(
    frame: pd.DataFrame, key: str, value: str, ids: tuple[str, ...]
) -> dict[str, float]:
    selected = frame.loc[frame[key].isin(ids)]
    if selected[key].duplicated().any() or set(selected[key]) != set(ids):
        raise ValueError(
            "A score table does not cover the fixed descriptor battery exactly once"
        )
    return {str(row[key]): float(row[value]) for row in selected.to_dict("records")}


def _assert_maps_close(
    first: dict[str, float], second: dict[str, float], label: str
) -> None:
    if set(first) != set(second) or not all(
        (np.isnan(first[key]) and np.isnan(second[key]))
        or np.isclose(first[key], second[key], rtol=0, atol=2e-14)
        for key in first
    ):
        raise ValueError(
            f"Recomputed {label} differs from the inherited point estimate"
        )


def _cell_inputs(
    workspace: Path,
    base_cfg: dict,
    analysis_cfg: dict,
    cell: dict,
    scope: str,
    shared_inputs: list[Path],
    frame_cache: dict[Path, pd.DataFrame],
) -> dict:
    source, target = cell["source"], cell["target"]
    pattern = analysis_cfg["inputs"]["evaluation_run_pattern"]
    inherited = workspace / analysis_cfg["inputs"]["inherited_protocol_run_root"]
    root = inherited / pattern.format(source=source, target=target, scope=scope)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["protocol_sha256"] != content_hash(base_cfg)
        or manifest["status"] != "complete"
    ):
        raise ValueError(
            "The inherited evaluation run is incomplete or from another protocol"
        )
    shared_inputs.append(manifest_path)

    observed_table = _verified_output(
        root,
        manifest,
        analysis_cfg["inputs"]["observed_target_point_table"],
        shared_inputs,
    )
    prediction_table = _verified_output(
        root, manifest, analysis_cfg["inputs"]["point_score_table"], shared_inputs
    )
    ranking_table = _verified_output(
        root, manifest, "secondary_selection_metrics.csv", shared_inputs
    )
    ids = tuple(observed_table.descriptor_id.astype(str))
    if len(ids) != len(set(ids)):
        raise ValueError("Inherited observed CCC table repeats a descriptor")
    observed_point = _mapping(observed_table, "descriptor_id", "ccc", ids)
    score_maps = {}
    for policy in ("source_ccc", "synthetic_ccc", "error_transport"):
        score_maps[policy] = _mapping(
            prediction_table.loc[prediction_table.policy.eq(policy)],
            "descriptor_id",
            "construction_score",
            ids,
        )
    repeated_observed = _mapping(
        prediction_table.loc[prediction_table.policy.eq("source_ccc")],
        "descriptor_id",
        "observed_target_ccc",
        ids,
    )
    _assert_maps_close(observed_point, repeated_observed, "observed target CCC")

    normalized_target = target.replace("\\", "/")
    reference_path = _input_path(
        manifest,
        lambda path: (
            str(path)
            .replace("\\", "/")
            .endswith(f"/{normalized_target}/reference_features.csv")
        ),
        shared_inputs,
    )
    alternate_path = _input_path(
        manifest,
        lambda path: (
            str(path)
            .replace("\\", "/")
            .endswith(f"/{normalized_target}/alternate_features.csv")
        ),
        shared_inputs,
    )
    split_path = _input_path(
        manifest, lambda path: path.name == "patient_split.csv", shared_inputs
    )
    for path in (reference_path, alternate_path, split_path):
        if path not in frame_cache:
            frame_cache[path] = pd.read_csv(
                path,
                float_precision="round_trip",
                dtype=str
                if path == split_path
                else {
                    "patient_id": str,
                    "unit_id": str,
                    "annotation_id": str,
                    "descriptor_id": str,
                },
                keep_default_na=path != split_path,
            )
    split = frame_cache[split_path]
    target_rows = split.loc[split.cohort.eq(target)]
    allowed = (
        set(target_rows.loc[target_rows.assignment.eq("audit"), "patient_id"])
        if scope == "audit"
        else set(target_rows.patient_id)
    )
    references = frame_cache[reference_path]
    alternates = frame_cache[alternate_path]
    references = references.loc[
        references.patient_id.isin(allowed) & references.descriptor_id.isin(ids)
    ].copy()
    alternates = alternates.loc[
        alternates.patient_id.isin(allowed) & alternates.descriptor_id.isin(ids)
    ].copy()
    observed_cache = source_pair_matrix(
        references, alternates, cohort=target, descriptor_ids=ids
    ).patient_moments()
    if set(observed_cache.patient_ids) != allowed:
        raise ValueError(
            "Observed target patient support differs from the locked split"
        )
    observed_original = aggregate_patient_moments(
        observed_cache,
        min_independent_patients=base_cfg["features"][
            "min_evaluation_independent_patients"
        ],
        min_weighted_support=base_cfg["features"]["min_evaluation_weighted_support"],
    )
    _assert_maps_close(
        observed_original.ccc_mapping(), observed_point, "observed target CCC"
    )

    synthetic_cache = synthetic_original = None
    if scope == "full":
        construction_scope = manifest["extra"]["construction_scope"]
        prefix = f"construct-{source}-to-{target}-{construction_scope}-"
        construction_manifest_path = _input_path(
            manifest,
            lambda path: (
                path.name == "run_manifest.json" and path.parent.name.startswith(prefix)
            ),
            shared_inputs,
        )
        construction_root = construction_manifest_path.parent
        construction_manifest = json.loads(
            construction_manifest_path.read_text(encoding="utf-8")
        )
        cache_name = analysis_cfg["inputs"]["full_synthetic_patient_cache"]
        cache_path = construction_root / cache_name
        if construction_manifest["outputs"].get(cache_name) != file_hash(cache_path):
            raise ValueError("The synthetic patient-moment cache is missing or changed")
        shared_inputs.append(cache_path)
        cache_frame = pd.read_csv(
            cache_path,
            float_precision="round_trip",
            dtype={"patient_id": str, "descriptor_id": str},
        )
        synthetic_cache = patient_cache_from_frame(cache_frame, descriptor_ids=ids)
        if synthetic_cache.patient_ids != observed_cache.patient_ids:
            raise ValueError(
                "Full observed and synthetic patient caches are not aligned"
            )
        synthetic_original = aggregate_patient_moments(
            synthetic_cache,
            min_independent_patients=1,
            min_weighted_support=base_cfg["features"][
                "min_construction_weighted_availability"
            ],
        )
        _assert_maps_close(
            synthetic_original.ccc_mapping(),
            score_maps["synthetic_ccc"],
            "synthetic target CCC",
        )

    return {
        "root": root,
        "source": source,
        "target": target,
        "scope": scope,
        "evidence_tier": cell["evidence_tier"],
        "ids": ids,
        "observed_point": observed_point,
        "score_maps": score_maps,
        "ranking_table": ranking_table,
        "observed_cache": observed_cache,
        "observed_original": observed_original,
        "synthetic_cache": synthetic_cache,
        "synthetic_original": synthetic_original,
        "patient_count": len(observed_cache.patient_ids),
    }


def _ranking_regression(
    cell: dict, preview_selection: pd.DataFrame, preview_anchors: pd.DataFrame
) -> list[dict]:
    screens = {item.policy: item for item in construct_screens(**cell["score_maps"])}
    inherited = cell["ranking_table"].set_index("policy")
    rows = []
    for policy, selection in screens.items():
        recomputed = evaluate_selection(selection, cell["observed_point"])
        old = inherited.loc[policy]
        selected_match = json.loads(old.selected_ids) == list(selection.selected_ids)
        values_match = all(
            (pd.isna(old[name]) and not isfinite(float(getattr(recomputed, name))))
            or np.isclose(
                float(old[name]), float(getattr(recomputed, name)), rtol=0, atol=2e-14
            )
            for name in (
                "utility",
                "top_quartile_precision",
                "observed_criterion_regret",
            )
        )
        key = (
            preview_selection.source.eq(cell["source"])
            & preview_selection.target.eq(cell["target"])
            & preview_selection.scope.eq(cell["scope"])
            & preview_selection.policy.eq(policy)
        )
        anchor_key = (
            preview_anchors.source.eq(cell["source"])
            & preview_anchors.target.eq(cell["target"])
            & preview_anchors.scope.eq(cell["scope"])
            & preview_anchors.policy.eq(policy)
        )
        if key.sum() != 1 or anchor_key.sum() != 1:
            raise ValueError(
                "The current supplementary ranking tables do not contain one row per cell and policy"
            )
        preview = preview_selection.loc[key].iloc[0]
        anchor = preview_anchors.loc[anchor_key].iloc[0]
        preview_match = np.isclose(
            float(preview.utility), recomputed.utility, rtol=0, atol=2e-14
        )
        anchor_match = np.isclose(
            float(anchor.U), recomputed.utility, rtol=0, atol=2e-14
        )
        passed = selected_match and values_match and preview_match and anchor_match
        rows.append(
            {
                "source_cohort": cell["source"],
                "target_cohort": cell["target"],
                "direction": f"{cell['source']}_to_{cell['target']}",
                "analysis_scope": cell["scope"],
                "policy": policy,
                "n_eligible": len(cell["ids"]),
                "budget": selection.budget,
                "inherited_utility": float(old.utility),
                "recomputed_utility": recomputed.utility,
                "inherited_top_quartile_precision": float(old.top_quartile_precision),
                "recomputed_top_quartile_precision": recomputed.top_quartile_precision,
                "inherited_observed_criterion_regret": float(
                    old.observed_criterion_regret
                ),
                "recomputed_observed_criterion_regret": recomputed.observed_criterion_regret,
                "selected_ids_exact_match": selected_match,
                "current_preview_utility_exact_match": preview_match,
                "current_anchor_utility_exact_match": anchor_match,
                "regression_passed": passed,
            }
        )
        if not passed:
            raise ValueError(
                "The preserved top-quarter analysis failed its exact regression check"
            )
    return rows


def _run_cell(
    cfg: dict, base_cfg: dict, cell: dict, descriptors: pd.DataFrame
) -> dict[str, list[dict]]:
    thresholds = tuple(
        float(value) for value in cfg["threshold_decision"]["thresholds"]
    )
    methods = {
        "synthetic_target": cell["score_maps"]["synthetic_ccc"],
        "observed_source": cell["score_maps"]["source_ccc"],
    }
    point, crossings = {}, []
    descriptor_metadata = descriptors.set_index("feature_id")
    for threshold in thresholds:
        point[threshold] = {
            method: threshold_decision(
                cell["observed_point"],
                scores,
                threshold=threshold,
                descriptor_ids=cell["ids"],
            )
            for method, scores in methods.items()
        }
        synthetic = point[threshold]["synthetic_target"]
        source = point[threshold]["observed_source"]
        for descriptor in cell["ids"]:
            meta = descriptor_metadata.loc[descriptor]
            real = cell["observed_point"][descriptor]
            syn = methods["synthetic_target"][descriptor]
            src = methods["observed_source"][descriptor]
            crossings.append(
                {
                    "analysis_scope": cell["scope"],
                    "evidence_tier": cell["evidence_tier"],
                    "source_cohort": cell["source"],
                    "target_cohort": cell["target"],
                    "direction": f"{cell['source']}_to_{cell['target']}",
                    "threshold": threshold,
                    "descriptor_id": descriptor,
                    "descriptor_name": meta["name"],
                    "descriptor_definition": meta["definition"],
                    "family": meta["family"],
                    "observed_target_ccc": real,
                    "synthetic_target_ccc": syn,
                    "observed_source_ccc": src,
                    "observed_target_retained": bool(real >= threshold)
                    if isfinite(real)
                    else pd.NA,
                    "synthetic_target_retained": bool(syn >= threshold)
                    if isfinite(syn)
                    else pd.NA,
                    "observed_source_retained": bool(src >= threshold)
                    if isfinite(src)
                    else pd.NA,
                    "synthetic_label": synthetic.labels[descriptor],
                    "source_label": source.labels[descriptor],
                    "synthetic_minus_observed_target_ccc": syn - real,
                    "source_minus_observed_target_ccc": src - real,
                }
            )

    requested = int(cfg["uncertainty"]["bootstrap_replicates"])
    counts = patient_bootstrap_multiplicities(
        cell["patient_count"],
        requested,
        rng=rng(
            cfg,
            "threshold_decision_bootstrap",
            cell["source"],
            cell["target"],
            cell["scope"],
        ),
    )
    values = {
        (threshold, method, metric): []
        for threshold in thresholds
        for method in methods
        for metric in ALL_METRICS
    }
    reasons = {key: [] for key in values}
    contrast_values = {
        (threshold, metric): []
        for threshold in thresholds
        for metric in CONTRAST_METRICS
    }
    contrast_reasons = {key: [] for key in contrast_values}
    for multiplicities in counts:
        observed = aggregate_patient_moments(
            cell["observed_cache"],
            multiplicities,
            locked_support=cell["observed_original"].support,
            min_weighted_support=base_cfg["features"][
                "min_evaluation_weighted_support"
            ],
        ).ccc_mapping()
        replicate_methods = dict(methods)
        if cell["scope"] == "full":
            replicate_methods["synthetic_target"] = aggregate_patient_moments(
                cell["synthetic_cache"],
                multiplicities,
                locked_support=cell["synthetic_original"].support,
                min_weighted_support=base_cfg["features"][
                    "min_construction_weighted_availability"
                ],
            ).ccc_mapping()
        for threshold in thresholds:
            results = {
                method: threshold_decision(
                    observed, scores, threshold=threshold, descriptor_ids=cell["ids"]
                )
                for method, scores in replicate_methods.items()
            }
            for method, result in results.items():
                for metric in ALL_METRICS:
                    values[(threshold, method, metric)].append(result.values[metric])
                    reasons[(threshold, method, metric)].append(result.reasons[metric])
            contrasts, why = decision_contrasts(
                results["synthetic_target"], results["observed_source"]
            )
            for metric in CONTRAST_METRICS:
                contrast_values[(threshold, metric)].append(contrasts[metric])
                contrast_reasons[(threshold, metric)].append(why[metric])

    metric_rows, interval_rows, contrast_rows = [], [], []
    for threshold in thresholds:
        for method in methods:
            result = point[threshold][method]
            summaries = {}
            for metric in ALL_METRICS:
                summary = summarize_replicates(
                    values[(threshold, method, metric)],
                    reasons[(threshold, method, metric)],
                    point=result.values[metric],
                    max_undefined_fraction=cfg["uncertainty"]["max_undefined_fraction"],
                    nominal_level=cfg["uncertainty"]["nominal_level"],
                )
                summaries[metric] = summary
                interval_rows.append(
                    {
                        "analysis_scope": cell["scope"],
                        "evidence_tier": cell["evidence_tier"],
                        "source_cohort": cell["source"],
                        "target_cohort": cell["target"],
                        "direction": f"{cell['source']}_to_{cell['target']}",
                        "threshold": threshold,
                        "method": method,
                        "metric": metric,
                        "n_eligible": len(cell["ids"]),
                        "conditioning": FULL_COHORT_CONDITIONING
                        if cell["scope"] == "full"
                        else AUDIT_CONDITIONING,
                        **{
                            key: json.dumps(value, sort_keys=True)
                            if key == "bootstrap_undefined_reasons"
                            else value
                            for key, value in summary.items()
                        },
                    }
                )
            statuses = {summary["interval_status"] for summary in summaries.values()}
            aggregate_status = (
                next(iter(statuses))
                if len(statuses) == 1
                else "mixed_see_interval_table"
            )
            metric_rows.append(
                {
                    "analysis_scope": cell["scope"],
                    "evidence_tier": cell["evidence_tier"],
                    "source_cohort": cell["source"],
                    "target_cohort": cell["target"],
                    "direction": f"{cell['source']}_to_{cell['target']}",
                    "threshold": threshold,
                    "method": method,
                    "n_eligible": len(cell["ids"]),
                    **result.values,
                    "bootstrap_replicates_requested": requested,
                    "bootstrap_replicates_defined": min(
                        summary["bootstrap_replicates_defined"]
                        for summary in summaries.values()
                    ),
                    "bootstrap_undefined_fraction": max(
                        summary["bootstrap_undefined_fraction"]
                        for summary in summaries.values()
                    ),
                    "interval_status": aggregate_status,
                    "conditioning": FULL_COHORT_CONDITIONING
                    if cell["scope"] == "full"
                    else AUDIT_CONDITIONING,
                }
            )
        point_contrast, point_reason = decision_contrasts(
            point[threshold]["synthetic_target"], point[threshold]["observed_source"]
        )
        for metric in CONTRAST_METRICS:
            summary = summarize_replicates(
                contrast_values[(threshold, metric)],
                contrast_reasons[(threshold, metric)],
                point=point_contrast[metric],
                max_undefined_fraction=cfg["uncertainty"]["max_undefined_fraction"],
                nominal_level=cfg["uncertainty"]["nominal_level"],
            )
            contrast_rows.append(
                {
                    "analysis_scope": cell["scope"],
                    "evidence_tier": cell["evidence_tier"],
                    "source_cohort": cell["source"],
                    "target_cohort": cell["target"],
                    "direction": f"{cell['source']}_to_{cell['target']}",
                    "threshold": threshold,
                    "metric": metric,
                    "sign_convention": "synthetic_minus_source",
                    "n_eligible": len(cell["ids"]),
                    "point_undefined_reason": point_reason[metric],
                    "conditioning": FULL_COHORT_CONDITIONING
                    if cell["scope"] == "full"
                    else AUDIT_CONDITIONING,
                    **{
                        key: json.dumps(value, sort_keys=True)
                        if key == "bootstrap_undefined_reasons"
                        else value
                        for key, value in summary.items()
                    },
                }
            )
    return {
        "metrics": metric_rows,
        "intervals": interval_rows,
        "contrasts": contrast_rows,
        "crossings": crossings,
    }


def run_threshold_experiment(
    workspace: Path, *, run_id: str, replicates: int | None = None
) -> Path:
    workspace = Path(workspace).resolve()
    config_path = workspace / "config/threshold_decision_fidelity_v1.yaml"
    base_path = workspace / "config/protocol.lock.yaml"
    amendment_path = workspace / "amendments/THRESHOLD_DECISION_FIDELITY_V1.md"
    cfg, base_cfg = read_protocol(config_path), read_protocol(base_path)
    if cfg["analysis"]["base_protocol_sha256"] != content_hash(base_cfg):
        raise ValueError(
            "The threshold experiment does not identify the active inherited protocol"
        )
    if replicates is not None:
        if replicates < 1:
            raise ValueError("A positive bootstrap replicate count is required")
        cfg = json.loads(json.dumps(cfg))
        cfg["uncertainty"]["bootstrap_replicates"] = int(replicates)
        cfg["analysis"]["status"] = (
            "timing_pilot" if replicates < 2000 else cfg["analysis"]["status"]
        )

    inputs = [
        config_path,
        base_path,
        amendment_path,
        workspace / "docs/14_CODEX_HANDOVER_THRESHOLD_DECISION_EXPERIMENT.md",
    ]
    descriptor_path = workspace / "audit/audited_descriptor_manifest.csv"
    inputs.append(descriptor_path)
    descriptors = pd.read_csv(descriptor_path, dtype=str, keep_default_na=False)
    preview_root = workspace / "reference/tables"
    preview_selection_path, preview_anchor_path = (
        preview_root / "selected_set_utility.csv",
        preview_root / "anchored_utility.csv",
    )
    inputs.extend([preview_selection_path, preview_anchor_path])
    preview_selection = pd.read_csv(
        preview_selection_path, float_precision="round_trip"
    )
    preview_anchors = pd.read_csv(preview_anchor_path, float_precision="round_trip")

    frame_cache, cells, ranking_rows = {}, [], []
    start = perf_counter()
    for declared in cfg["threshold_decision"]["directions"]:
        for scope in cfg["threshold_decision"]["analysis_scopes"]:
            cell = _cell_inputs(
                workspace, base_cfg, cfg, declared, scope, inputs, frame_cache
            )
            cells.append(cell)
            ranking_rows.extend(
                _ranking_regression(cell, preview_selection, preview_anchors)
            )

    all_rows = {name: [] for name in ("metrics", "intervals", "contrasts", "crossings")}
    for cell in cells:
        outputs = _run_cell(cfg, base_cfg, cell, descriptors)
        for name in all_rows:
            all_rows[name].extend(outputs[name])

    metrics, intervals = (
        pd.DataFrame(all_rows["metrics"]),
        pd.DataFrame(all_rows["intervals"]),
    )
    contrasts, crossings = (
        pd.DataFrame(all_rows["contrasts"]),
        pd.DataFrame(all_rows["crossings"]),
    )
    thresholds = list(map(float, cfg["threshold_decision"]["thresholds"]))
    if sorted(metrics.threshold.unique()) != thresholds:
        raise ValueError("Threshold output grid differs from the locked specification")
    group = metrics.sort_values("threshold").groupby(
        ["direction", "analysis_scope", "method"]
    )
    if any((frame.n_method_retained.diff().dropna() > 0).any() for _, frame in group):
        raise ValueError("Retained-set size increased as its CCC cutoff increased")
    if not pd.DataFrame(ranking_rows).regression_passed.all():
        raise ValueError("The ranking sensitivity regression did not pass")
    expected_crossings = sum(len(cell["ids"]) for cell in cells) * len(thresholds)
    if (
        len(crossings) != expected_crossings
        or crossings.duplicated(
            ["direction", "analysis_scope", "threshold", "descriptor_id"]
        ).any()
    ):
        raise ValueError(
            "The descriptor crossing table is incomplete or has repeated identities"
        )

    primary = intervals.loc[
        intervals.evidence_tier.eq("principal_mri")
        & intervals.metric.isin(
            ["false_retention_proportion", "false_exclusion_proportion", "jaccard"]
        )
    ].copy()
    count_columns = [
        "direction",
        "analysis_scope",
        "threshold",
        "method",
        "n_target_retained",
        "n_method_retained",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    figure_source = primary.merge(
        metrics[count_columns],
        on=["direction", "analysis_scope", "threshold", "method"],
        validate="many_to_one",
    )

    writer = RunWriter(
        workspace,
        cfg,
        "threshold_decision_fidelity",
        run_id,
        inputs=list(dict.fromkeys(inputs)),
        estimand="ccc_threshold_decision_fidelity_relative_to_observed_target_reference_decision",
        role="review_driven_experiment3_reanalysis_of_existing_measurements",
        seeds={
            "stream": "threshold_decision_bootstrap",
            "keys": ["source", "target", "scope"],
        },
    )
    writer.write_table("experiment3_threshold_decision_metrics.csv", metrics)
    writer.write_table("experiment3_threshold_decision_intervals.csv", intervals)
    writer.write_table("experiment3_threshold_decision_contrasts.csv", contrasts)
    writer.write_table("experiment3_threshold_descriptor_crossings.csv", crossings)
    writer.write_table("experiment3_threshold_figure_source.csv", figure_source)
    writer.write_table(
        "experiment3_ranking_sensitivity_regression.csv", pd.DataFrame(ranking_rows)
    )
    writer.write_json(
        "experiment3_threshold_analysis_spec.json",
        {
            "analysis_config_sha256": content_hash(cfg),
            "base_protocol_sha256": content_hash(base_cfg),
            "thresholds": thresholds,
            "retain_rule": cfg["threshold_decision"]["retain_rule"],
            "bootstrap_replicates": cfg["uncertainty"]["bootstrap_replicates"],
            "ccc_implementation": "redesign2.moments._joint_from_components/ccc",
            "eligible_counts": {
                f"{cell['source']}_to_{cell['target']}:{cell['scope']}": len(
                    cell["ids"]
                )
                for cell in cells
            },
            "new_contour_generation": False,
            "new_image_preprocessing": False,
            "ranking_sensitivity_regression_passed": True,
        },
    )
    elapsed = perf_counter() - start
    return writer.finish(
        {
            "directions": len(cfg["threshold_decision"]["directions"]),
            "analysis_scopes": ["full", "audit"],
            "thresholds": thresholds,
            "bootstrap_replicates_per_cell": cfg["uncertainty"]["bootstrap_replicates"],
            "elapsed_seconds": elapsed,
            "new_contour_generation": False,
            "new_image_preprocessing": False,
            "ranking_sensitivity_regression_passed": True,
        }
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replicates", type=int)
    args = parser.parse_args(argv)
    print(
        run_threshold_experiment(
            args.workspace, run_id=args.run_id, replicates=args.replicates
        )
    )


if __name__ == "__main__":
    main()
