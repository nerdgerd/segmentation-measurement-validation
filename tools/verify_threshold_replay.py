"""Compare a newly executed threshold run with every recorded numerical output.

The locked threshold specification also covers cohorts outside the reported study,
so a fresh run contains their rows. Only the reported MRI cohorts are compared, and
the recorded tables contain exactly those rows (provenance/CT_REMOVAL.json).
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTED_COHORTS = {"upenn", "brats_tcga"}


def reported_rows(path):
    """Keep the header and reported-cohort records as raw text, then parse them.

    Parsing after selection gives each column the type it has in the recorded
    table; a column can hold text only in the rows of the other cohorts.
    """
    text = path.read_text(encoding="utf-8")
    records, start, quoted = [], 0, False
    for index, char in enumerate(text):
        if char == '"':
            quoted = not quoted
        elif char == "\n" and not quoted:
            records.append(text[start : index + 1])
            start = index + 1
    if start < len(text):
        records.append(text[start:])
    cohorts = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if len(records) != len(cohorts) + 1:
        raise ValueError(f"{path.name}: records do not match parsed rows")
    keep = cohorts.source_cohort.isin(REPORTED_COHORTS) & cohorts.target_cohort.isin(
        REPORTED_COHORTS
    )
    kept = records[0] + "".join(r for r, k in zip(records[1:], keep) if k)
    return pd.read_csv(io.StringIO(kept), float_precision="round_trip")
NAMES = [
    "experiment3_threshold_decision_metrics.csv",
    "experiment3_threshold_decision_intervals.csv",
    "experiment3_threshold_decision_contrasts.csv",
    "experiment3_threshold_descriptor_crossings.csv",
    "experiment3_threshold_figure_source.csv",
    "experiment3_ranking_sensitivity_regression.csv",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    count = 0
    for name in NAMES:
        actual = reported_rows(args.run / name)
        expected = pd.read_csv(
            ROOT / "study/reference/tables" / name, float_precision="round_trip"
        )
        if (
            actual.columns.tolist() != expected.columns.tolist()
            or actual.shape != expected.shape
        ):
            raise AssertionError(f"Schema/support changed: {name}")
        for column in expected.columns:
            if column == "run_id":
                continue
            if pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(
                    actual[column],
                    expected[column],
                    rtol=0,
                    atol=2e-14,
                    equal_nan=True,
                    err_msg=f"{name}:{column}",
                )
            else:
                pd.testing.assert_series_equal(
                    actual[column], expected[column], check_names=True
                )
            count += 1
    print(
        json.dumps(
            {
                "tables": len(NAMES),
                "column_checks": count,
                "failures": 0,
                "compared": "all numerical outputs and metadata except the new storage run_id",
            }
        )
    )


if __name__ == "__main__":
    main()
