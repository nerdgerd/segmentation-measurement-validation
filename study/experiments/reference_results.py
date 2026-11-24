"""Direct manuscript summaries and point-decision replay from immutable tables."""

from pathlib import Path
import numpy as np
import pandas as pd
from redesign2.threshold_decisions import threshold_decision


def table(reference, name):
    return pd.read_csv(
        Path(reference) / "tables" / (name + ".csv"), float_precision="round_trip"
    )


def verify_threshold_points(reference):
    crossings = table(reference, "experiment3_threshold_descriptor_crossings")
    metrics = table(reference, "experiment3_threshold_decision_metrics")
    cells = 0
    for identity, rows in crossings.groupby(
        ["direction", "analysis_scope", "threshold"]
    ):
        observed = dict(zip(rows.descriptor_id, rows.observed_target_ccc, strict=True))
        for method, column in [
            ("synthetic_target", "synthetic_target_ccc"),
            ("observed_source", "observed_source_ccc"),
        ]:
            scores = dict(zip(rows.descriptor_id, rows[column], strict=True))
            actual = threshold_decision(observed, scores, threshold=float(identity[2]))
            expected = metrics[
                (metrics.direction == identity[0])
                & (metrics.analysis_scope == identity[1])
                & (metrics.threshold == identity[2])
                & (metrics.method == method)
            ].iloc[0]
            for name, value in actual.values.items():
                if not np.isclose(
                    value, expected[name], atol=2e-14, rtol=0, equal_nan=True
                ):
                    raise AssertionError(
                        f"Changed cutoff decision: {identity}/{method}/{name}"
                    )
            cells += 1
    return cells


def manuscript_summary(reference):
    profiles = table(reference, "profile_agreement")
    differences = table(reference, "descriptor_disagreement")
    population = table(reference, "matched_population_association")
    cutoffs = table(reference, "experiment3_threshold_decision_metrics")
    records = []
    for row in profiles[
        (profiles.policy == "synthetic_ccc")
        & profiles.target.isin(["upenn", "brats_tcga"])
    ].itertuples():
        baseline = population[
            (population.target == row.target)
            & (population.scope == row.scope)
            & (population.battery == "complete")
            & (population.theta == 0)
        ]
        ratios = differences[
            (differences.target == row.target)
            & (differences.scope == row.scope)
            & (differences.policy == "synthetic_ccc")
        ].log2_rmse_ratio
        threshold = cutoffs[
            (cutoffs.direction == f"{row.source}_to_{row.target}")
            & (cutoffs.analysis_scope == row.scope)
            & (cutoffs.method == "synthetic_target")
            & (cutoffs.threshold == 0.95)
        ].iloc[0]
        records.append(
            {
                "direction": f"{row.source} -> {row.target}",
                "patients": "all" if row.scope == "full" else "held-out",
                "ranking_correlation": row.profile_spearman,
                "simulation_baseline": float(baseline.profile_spearman.median()),
                "median_log2_rmse_ratio": float(ratios.median()),
                "median_absolute_log2_rmse_ratio": float(ratios.abs().median()),
                "falsely_kept_at_095": int(threshold.fp),
                "falsely_discarded_at_095": int(threshold.fn),
            }
        )
    return records
