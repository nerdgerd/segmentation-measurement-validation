"""Report all saved ranking endpoints and their undefinedness."""

from __future__ import annotations
from redesign2.execution import read_text
from pathlib import Path
from redesign2.execution import copy_input
import json
import numpy as np
import pandas as pd
from redesign2.provenance import RunWriter, read_protocol
from redesign2.execution import read_csv, storage_run_id


def run(workspace, run_id):
    """Report all saved ranking endpoints and their undefinedness."""
    W = Path(workspace).resolve()
    C = W / "critical_revision"
    cfg = read_protocol(W / "config/measurement_validation_amendment_v2.yaml")
    A = W / json.loads(read_text(C / "analysis_package.json"))["run"]
    O = (
        W
        / "runs"
        / cfg["revision"]["inherited_protocol_sha256"]
        / "study-report-unitprecision-v2"
    )
    trace = read_csv(A / "bootstrap_trace.csv")
    original = read_csv(O / "bootstrap_replicates.csv")
    keys = ["source", "target", "scope", "replicate"]
    complete = trace[trace.battery.eq("complete")]
    joined = complete.merge(
        original, on=keys, validate="one_to_one", suffixes=("_replay", "_original")
    )
    for p in ["source_ccc", "synthetic_ccc", "error_transport"]:
        assert np.allclose(
            joined["utility:" + p + "_replay"],
            joined["utility:" + p + "_original"],
            equal_nan=True,
            rtol=1e-10,
            atol=1e-10,
        )
    contrast = original["utility:synthetic_ccc"] - original["utility:source_ccc"]
    expected = original[keys].copy()
    expected["contrast"] = contrast
    check = complete.merge(expected, on=keys, validate="one_to_one")
    assert np.allclose(
        check.synthetic_minus_source,
        check.contrast,
        equal_nan=True,
        rtol=1e-10,
        atol=1e-10,
    )
    rows = []
    for key, g in trace.groupby(["source", "target", "scope", "battery"]):
        for endpoint in [
            "utility:source_ccc",
            "utility:synthetic_ccc",
            "utility:error_transport",
            "synthetic_minus_source",
        ]:
            values = g[endpoint].to_numpy()
            defined = values[np.isfinite(values)]
            fraction = 1 - len(defined) / len(values)
            issued = fraction <= 0.05 + 1e-15
            rows.append(
                dict(
                    zip(["source", "target", "scope", "battery"], key),
                    endpoint=endpoint,
                    replicates=len(g),
                    defined=len(defined),
                    undefined_replicate_fraction=fraction,
                    range_issued=issued,
                    lower=float(np.quantile(defined, 0.025))
                    if issued and len(defined)
                    else np.nan,
                    upper=float(np.quantile(defined, 0.975))
                    if issued and len(defined)
                    else np.nan,
                )
            )
    reason = read_csv(A / "undefined_reason_counts.csv")
    vocabulary = json.loads(read_text(A / "reason_vocabulary.json"))["codes"]
    causes = []
    for key, g in reason.groupby(["source", "target", "scope", "battery", "policy"]):
        endpoints = g[g.descriptor_id.isna()]
        assert endpoints.replicates.sum() == 2000
        for code in ["defined", *vocabulary]:
            exclusive = int(
                endpoints.loc[endpoints.reason.eq(code), "replicates"].sum()
            )
            descriptors = int(
                g.loc[
                    g.descriptor_id.notna() & g.detail_code.eq(code), "replicates"
                ].sum()
            )
            causes.append(
                dict(
                    zip(["source", "target", "scope", "battery", "policy"], key),
                    reason=code,
                    exclusive_endpoint_count=exclusive,
                    overlapping_descriptor_cause_count=descriptors,
                )
            )
    inputs = [
        Path(__file__),
        C / "AMENDMENT_LOCK.json",
        A / "run_manifest.json",
        A / "bootstrap_trace.csv",
        A / "undefined_reason_counts.csv",
        A / "reason_vocabulary.json",
        O / "bootstrap_replicates.csv",
    ]
    writer = RunWriter(
        W,
        cfg,
        "critical_diagnostic_summaries",
        storage_run_id(run_id, "diagnostic-summary-v1"),
        inputs=inputs,
        role="all_original_endpoint_replays_and_zero_count_reason_codes",
        estimand="descriptive_definedness_and_sampling_stability",
    )
    writer.write_table("bootstrap_definedness_summary.csv", pd.DataFrame(rows))
    writer.write_table("bootstrap_reason_vocabulary_counts.csv", pd.DataFrame(causes))
    copy_input(__file__, writer.temp / "summary_source.py")
    final = writer.finish()
    (C / "diagnostic_package.json").write_text(
        json.dumps(dict(run=str(final.relative_to(W)).replace("\\", "/")), indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "Verified all original utilities and S--D replays; summarized",
        len(rows),
        "endpoints",
    )
