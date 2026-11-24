"""Attach recorded guard reasons to each saved bootstrap endpoint."""

from __future__ import annotations
from redesign2.execution import read_text
from pathlib import Path
import json
import numpy as np
import pandas as pd
from redesign2.provenance import RunWriter, read_protocol, verify_run
from redesign2.execution import read_csv, storage_run_id


def run(workspace, run_id):
    """Attach recorded guard reasons to each saved bootstrap endpoint."""
    W = Path(workspace).resolve()
    C = W / "critical_revision"
    A = W / json.loads(read_text(C / "analysis_package.json"))["run"]
    assert not verify_run(A)
    trace = read_csv(A / "bootstrap_trace.csv", float_precision="round_trip")
    failed = read_csv(A / "undefined_replicate_reasons.csv")
    keys = ["source", "target", "scope", "battery", "replicate"]
    policies = ["source_ccc", "synthetic_ccc", "error_transport"]
    frames = []
    assert not failed.duplicated(keys + ["policy"]).any()
    for policy in policies:
        frame = (
            trace[keys + ["utility:" + policy]]
            .rename(columns={"utility:" + policy: "value"})
            .copy()
        )
        frame["policy"] = policy
        frame = frame.merge(
            failed[keys + ["policy", "reason"]],
            on=keys + ["policy"],
            how="left",
            validate="one_to_one",
        )
        defined = np.isfinite(frame.value)
        assert (
            frame.loc[~defined, "reason"].notna().all()
            and frame.loc[defined, "reason"].isna().all()
        )
        frame.loc[defined, "reason"] = "defined"
        frames.append(frame)
    left = frames[0][keys + ["reason"]].rename(columns={"reason": "source_reason"})
    right = frames[1][keys + ["reason"]].rename(columns={"reason": "synthetic_reason"})
    contrast = (
        trace[keys + ["synthetic_minus_source"]]
        .rename(columns={"synthetic_minus_source": "value"})
        .merge(left, on=keys, validate="one_to_one")
        .merge(right, on=keys, validate="one_to_one")
    )
    contrast["policy"] = "synthetic_minus_source"
    contrast["reason"] = [
        "|".join(sorted({r for r in [s, t] if r != "defined"})) or "defined"
        for s, t in zip(contrast.source_reason, contrast.synthetic_reason)
    ]
    assert np.array_equal(np.isfinite(contrast.value), contrast.reason.eq("defined"))
    frames.append(contrast[keys + ["policy", "value", "reason"]])
    complete = pd.concat(frames, ignore_index=True).sort_values(keys + ["policy"])
    assert len(complete) == 128000 and (
        not complete.duplicated(keys + ["policy"]).any()
    )
    summary = (
        complete.groupby(
            ["source", "target", "scope", "battery", "policy", "reason"], dropna=False
        )
        .size()
        .rename("replicates")
        .reset_index()
    )
    writer = RunWriter(
        W,
        read_protocol(W / "config/measurement_validation_amendment_v2.yaml"),
        "complete_bootstrap_reason_records",
        storage_run_id(run_id, "bootstrap-reasons-v1"),
        inputs=[
            Path(__file__),
            C / "AMENDMENT_LOCK.json",
            A / "run_manifest.json",
            A / "bootstrap_trace.csv",
            A / "undefined_replicate_reasons.csv",
        ],
        role="every_saved_endpoint_guard_and_derived_contrast_union",
        estimand="unchanged_original_and_denominator_sensitivity_bootstrap_utilities",
    )
    writer.write_table("complete_bootstrap_reasons.csv", complete)
    writer.write_table("complete_bootstrap_reason_summary.csv", summary)
    writer.write_json(
        "contrast_reason_contract.json",
        {
            "defined": "both synthetic and source utilities finite",
            "undefined": "sorted union of the saved synthetic/source guard codes",
            "numerical_values": "copied from preserved bootstrap trace, not recomputed",
        },
    )
    final = writer.finish()
    assert not verify_run(final)
    (C / "bootstrap_reason_package.json").write_text(
        json.dumps({"run": str(final.relative_to(W)).replace("\\", "/")}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Complete endpoint reason records:", len(complete))
