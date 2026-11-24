"""Summarize reference-variance differences by feature family and region."""

from __future__ import annotations
from redesign2.execution import read_text
from pathlib import Path
from redesign2.execution import copy_input
import json
import pandas as pd
from redesign2.provenance import RunWriter, read_protocol
from redesign2.mechanism import distribution
from redesign2.execution import read_csv, storage_run_id


def run(workspace, run_id):
    """Summarize reference-variance differences by feature family and region."""
    W = Path(workspace).resolve()
    C = W / "critical_revision"
    A = W / json.loads(read_text(C / "analysis_package.json"))["run"]
    source = A / "case_mix_variance_shift.csv"
    d = read_csv(source)
    rows = []
    keys = ["source", "target", "scope", "family", "region"]
    for key, g in d.groupby(keys, dropna=False, sort=True):
        rows.append(
            dict(
                zip(keys, key),
                **distribution(g.log2_reference_variance_ratio),
                minimum=float(g.log2_reference_variance_ratio.min()),
                maximum=float(g.log2_reference_variance_ratio.max()),
            )
        )
    writer = RunWriter(
        W,
        read_protocol(W / "config/measurement_validation_amendment_v2.yaml"),
        "case_mix_strata",
        storage_run_id(run_id, "case-mix-strata-v1"),
        inputs=[
            Path(__file__),
            source,
            A / "run_manifest.json",
            C / "AMENDMENT_LOCK.json",
        ],
        role="every_family_region_and_direction_retained",
        estimand="descriptive_reference_variance_shift",
    )
    writer.write_table("case_mix_family_region_summary.csv", pd.DataFrame(rows))
    copy_input(__file__, writer.temp / "summary_source.py")
    final = writer.finish()
    (C / "case_mix_package.json").write_text(
        json.dumps({"run": str(final.relative_to(W)).replace("\\", "/")}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Case-mix strata", len(rows))
