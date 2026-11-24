"""Report zero means and continuous mean-relative-to-SD diagnostics."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from redesign2.provenance import RunWriter, read_protocol, verify_run
from redesign2.execution import read_csv, storage_run_id


def run(workspace, run_id):
    """Report zero means and continuous mean-relative-to-SD diagnostics."""
    W = Path(workspace).resolve()
    C = W / "critical_revision"
    cfg = read_protocol(W / "config/measurement_validation_amendment_v2.yaml")
    R = (
        W
        / "runs"
        / cfg["revision"]["inherited_protocol_sha256"]
        / "study-report-unitprecision-v2"
    )
    source = R / "reference_evaluation_moments.csv"
    d = read_csv(source, float_precision="round_trip")
    definition = read_csv(C / "descriptor_definition_audit.csv")
    flags = [
        "region",
        "natural_zero",
        "ratio_or_CV",
        "angular",
        "signed",
        "denominator_risk",
    ]
    assert definition.groupby("descriptor_id")[flags].nunique().le(1).all().all()
    d = d.merge(
        definition.drop_duplicates("descriptor_id")[["descriptor_id", *flags]],
        on="descriptor_id",
        validate="many_to_one",
    )
    assert np.isfinite(d.reference_mean).all() and (d.reference_variance >= 0).all()
    d["reference_SD"] = np.sqrt(d.reference_variance)
    d["exact_zero_reference_mean"] = d.reference_mean.eq(0)
    d["absolute_mean_over_SD"] = np.divide(
        abs(d.reference_mean),
        d.reference_SD,
        out=np.full(len(d), np.nan),
        where=d.reference_SD.gt(0),
    )
    rows = []
    for key, g in d.groupby(["source", "target", "scope"]):
        a = g.absolute_mean_over_SD[np.isfinite(g.absolute_mean_over_SD)]
        rows.append(
            dict(
                zip(["source", "target", "scope"], key),
                descriptors=len(g),
                exact_zero_means=int(g.exact_zero_reference_mean.sum()),
                zero_reference_variances=int(g.reference_SD.eq(0).sum()),
                minimum=float(a.min()),
                q25=float(a.quantile(0.25)),
                median=float(a.median()),
                q75=float(a.quantile(0.75)),
            )
        )
    writer = RunWriter(
        W,
        cfg,
        "continuous_zero_point_diagnostics",
        storage_run_id(run_id, "zero-point-diagnostics-v1"),
        inputs=[
            Path(__file__),
            C / "ZERO_POINT_DIAGNOSTICS_RULE.md",
            C / "AMENDMENT_LOCK.json",
            C / "descriptor_definition_audit.csv",
            source,
            R / "run_manifest.json",
        ],
        role="all_original_eligible_ids_no_threshold_no_new_exclusion",
        estimand="exact_zero_mean_and_continuous_reference_mean_relative_to_SD",
    )
    writer.write_table("zero_point_descriptor_diagnostics.csv", d)
    writer.write_table("zero_point_summary.csv", pd.DataFrame(rows))
    final = writer.finish()
    assert not verify_run(final)
    (C / "zero_point_package.json").write_text(
        json.dumps({"run": str(final.relative_to(W)).replace("\\", "/")}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Continuous zero-point diagnostic rows:", len(d))
