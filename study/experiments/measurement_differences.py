"""Question 2: replay every RMSE ratio from recorded mean squared differences."""

from pathlib import Path
from decimal import Decimal
from math import log2
import csv
import numpy as np
import pandas as pd

from redesign2.moments import measurement_difference_ratio
from redesign2.provenance import RunWriter, read_protocol


def replay(reference):
    path = Path(reference) / "tables/descriptor_disagreement.csv"
    recorded = pd.read_csv(path, float_precision="round_trip")
    rows = recorded[recorded.policy.eq("synthetic_ccc")].copy()
    computed = [
        measurement_difference_ratio(
            row.predicted_disagreement_mse, row.observed_disagreement_mse
        )
        for row in rows.itertuples()
    ]
    values = np.array([value for value, _ in computed])
    # Earlier table staging shortened some small M columns through decimal
    # parsing. Check their literal decimal precision rather than retuning L.
    with path.open(encoding="utf-8", newline="") as handle:
        strings = [
            row for row in csv.DictReader(handle) if row["policy"] == "synthetic_ccc"
        ]
    for index, row in enumerate(strings):
        if row["rmse_stratum"] != "finite":
            if not np.isclose(
                values[index], rows.log2_rmse_ratio.iloc[index], equal_nan=True
            ):
                raise AssertionError("A zero/undefined RMSE ratio changed")
            continue
        intervals = []
        for column in ["predicted_disagreement_mse", "observed_disagreement_mse"]:
            number = Decimal(row[column])
            quantum = Decimal(10) ** number.as_tuple().exponent
            lower, upper = float(number - quantum), float(number + quantum)
            if lower <= 0:
                raise ValueError(
                    "Saved M precision does not determine a finite log ratio"
                )
            intervals.append((lower, upper))
        low = 0.5 * (log2(intervals[0][0]) - log2(intervals[1][1]))
        high = 0.5 * (log2(intervals[0][1]) - log2(intervals[1][0]))
        rounding = 16 * np.finfo(float).eps * max(1, abs(low), abs(high))
        if not low - rounding <= rows.log2_rmse_ratio.iloc[index] <= high + rounding:
            raise AssertionError(
                "An RMSE ratio falls outside the saved M columns' decimal precision"
            )
    if [stratum for _, stratum in computed] != rows.rmse_stratum.tolist():
        raise AssertionError("A zero/undefined RMSE stratum changed")
    rows["recomputed_log2_rmse_ratio"] = values
    return rows


def run(workspace, run_id):
    workspace = Path(workspace).resolve()
    reference = workspace / "reference"
    rows = replay(reference)
    summaries = []
    for identity, group in rows.groupby(["source", "target", "scope"]):
        finite = group.recomputed_log2_rmse_ratio[
            np.isfinite(group.recomputed_log2_rmse_ratio)
        ]
        summaries.append(
            dict(
                zip(["source", "target", "scope"], identity),
                eligible_features=len(group),
                finite_ratios=len(finite),
                median_L=float(finite.median()),
                median_absolute_L=float(finite.abs().median()),
            )
        )
    writer = RunWriter(
        workspace,
        read_protocol(workspace / "config/protocol.lock.yaml"),
        "measurement_difference_replay",
        run_id,
        inputs=[
            Path(__file__),
            reference / "tables/descriptor_disagreement.csv",
            workspace / "src/redesign2/moments.py",
        ],
        estimand="recorded_single_contour_generated_to_original_RMSE_ratio",
        role="reference_result_replay_from_recorded_squared_measurement_differences",
    )
    writer.write_table("measurement_difference_feature_replay.csv", rows)
    writer.write_table("measurement_difference_summary.csv", pd.DataFrame(summaries))
    result = writer.finish()
    print(result)
    return result
