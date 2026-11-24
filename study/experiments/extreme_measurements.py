"""Trace the largest measurement ratios to their saved contour measurements."""

from __future__ import annotations
from redesign2.execution import read_text
from pathlib import Path
from redesign2.execution import copy_input
import json
import re
import numpy as np
import pandas as pd
from redesign2.provenance import (
    RunWriter,
    read_protocol,
    content_hash,
    file_hash,
    verify_run,
)
from redesign2.execution import read_csv, storage_run_id


def run(workspace, run_id):
    """Trace the largest measurement ratios to their saved contour measurements."""
    WORK = Path(workspace).resolve()
    REV = WORK / "media_revision"
    ANALYSIS = (
        WORK
        / json.loads(read_text(REV / "analysis_package.json", encoding="utf-8"))["run"]
    )
    OLD = WORK / "runs/8139ac7fc83dcba4a345f3b76636bacede3497c3d503d19e32d05d9083504a8d"
    base = read_protocol(WORK / "config/media_revision_20260917.yaml")
    cfg = dict(base)
    cfg["revision"] = dict(
        base["revision"],
        id="media_revision_extreme_trace_20260917",
        parent_revision_sha256=content_hash(base),
    )
    cfg["extreme_trace"] = {
        "selection": "all_volume_ratio_identities_in_locked_top_ten_bias_slope_audit",
        "epsilon_voxel_count": 1e-06,
        "audit_split": "original_locked_metadata",
        "random_sampling": False,
        "endpoint_change": False,
        "all_original_values_retained": True,
    }
    configpath = WORK / "config/media_revision_extreme_trace_20260917.yaml"
    if content_hash(read_protocol(configpath)) != content_hash(cfg):
        raise ValueError("The recorded extreme-measurement specification changed")
    oldcfg = read_protocol(WORK / "config/protocol.lock.yaml")
    splitpath = WORK / oldcfg["resolved_metadata"]["patient_split"]
    inputpaths = [
        ANALYSIS / "extreme_descriptor_diagnosis.csv",
        WORK / "MEDIA_REVISION_EXTREME_TRACE_ADDENDUM.md",
        configpath,
        WORK / "src/redesign2/adapters/descriptors.py",
        splitpath,
        Path(__file__),
    ]
    inputpaths += [
        OLD / "prepared-20260915" / c / (role + "_features.csv")
        for c in ["upenn", "brats_tcga"]
        for role in ["reference", "alternate"]
    ]
    writer = RunWriter(
        WORK,
        cfg,
        "media_revision_extreme_trace",
        storage_run_id(run_id, "retained-volume-ratio-trace-v1"),
        inputs=inputpaths,
        role="explanatory_trace_no_changed_scientific_estimates",
        estimand="retained_contour_measurements_and_disagreement_covariance",
        seeds={"random_sampling": "none"},
    )
    ex = read_csv(ANALYSIS / "extreme_descriptor_diagnosis.csv")
    ex = ex[ex.descriptor_id.str.startswith("volratio_")].drop_duplicates(
        ["source", "target", "scope", "descriptor_id"]
    )
    split = read_csv(splitpath)
    loaded = {
        c: {
            role: read_csv(OLD / "prepared-20260915" / c / (role + "_features.csv"))
            for role in ["reference", "alternate"]
        }
        for c in ["upenn", "brats_tcga"]
    }
    summaries = []
    records = []
    for _, r in ex.iterrows():
        numerator, denominator = re.fullmatch(
            "volratio_(.+)_over_(.+)", r.descriptor_id
        ).groups()
        paired = {}
        for role in ["reference", "alternate"]:
            g = loaded[r.target][role]
            if r.scope == "audit":
                pts = split[
                    split.cohort.eq(r.target) & split.assignment.eq("audit")
                ].patient_id
                g = g[g.patient_id.isin(pts)]
            ids = [
                r.descriptor_id,
                "vol_" + numerator + "_cm3",
                "vol_" + denominator + "_cm3",
            ]
            g = g[g.descriptor_id.isin(ids)]
            assert g.definedness.eq("defined").all() and np.isfinite(g.value).all()
            pivot = g.pivot(
                index=["patient_id", "unit_id", "annotation_id"],
                columns="descriptor_id",
                values="value",
            )
            assert (
                pivot.notna().all().all()
                and pivot.reset_index().groupby("patient_id").size().eq(1).all()
            )
            pivot = pivot.reset_index().set_index("patient_id").sort_index()
            paired[role] = pivot
            n = pivot["vol_" + numerator + "_cm3"] * 1000
            d = pivot["vol_" + denominator + "_cm3"] * 1000
            ratio = n / (d + 1e-06)
            assert np.allclose(ratio, pivot[r.descriptor_id], rtol=1e-12, atol=1e-12)
            for patient, row in pivot.iterrows():
                records.append(
                    dict(
                        source=r.source,
                        target=r.target,
                        scope=r.scope,
                        descriptor_id=r.descriptor_id,
                        patient_id=patient,
                        unit_id=row.unit_id,
                        annotation_id=row.annotation_id,
                        role=role,
                        ratio=row[r.descriptor_id],
                        numerator_voxels=row["vol_" + numerator + "_cm3"] * 1000,
                        denominator_voxels=row["vol_" + denominator + "_cm3"] * 1000,
                        epsilon_voxels=1e-06,
                        definedness="defined",
                    )
                )
        ref, alt = (paired["reference"], paired["alternate"])
        assert ref.index.equals(alt.index)
        x = ref[r.descriptor_id].to_numpy()
        w = alt[r.descriptor_id].to_numpy()
        delta = w - x
        v = np.mean((x - x.mean()) ** 2)
        a = np.mean((x - x.mean()) * (delta - delta.mean()))
        zero = alt["vol_" + denominator + "_cm3"].eq(0).to_numpy()
        contribution = np.sum(((x - x.mean()) * (delta - delta.mean()))[zero]) / len(x)
        assert np.allclose(
            [v, a],
            [r.reference_variance, r.observed_covariance],
            rtol=1e-12,
            atol=1e-10,
        )
        summaries.append(
            dict(
                source=r.source,
                target=r.target,
                scope=r.scope,
                descriptor_id=r.descriptor_id,
                numerator=numerator,
                denominator=denominator,
                patients=len(x),
                reference_mean=x.mean(),
                reference_variance=v,
                reference_cv_squared=v / x.mean() ** 2,
                observed_covariance=a,
                zero_reference_denominator=int(
                    ref["vol_" + denominator + "_cm3"].eq(0).sum()
                ),
                zero_alternate_denominator=int(zero.sum()),
                zero_alternate_covariance_contribution=contribution,
                zero_alternate_covariance_fraction=contribution / a if a else np.nan,
                max_reference_ratio=float(x.max()),
                max_alternate_ratio=float(w.max()),
                min_alternate_denominator_voxels=float(
                    (alt["vol_" + denominator + "_cm3"] * 1000).min()
                ),
                inherited_denominator_epsilon_voxels=1e-06,
                near_constant_reference_flag=bool(v / x.mean() ** 2 < 1e-06),
            )
        )
    writer.write_table("volume_ratio_extreme_trace.csv", pd.DataFrame(summaries))
    writer.write_table("volume_ratio_contour_records.csv", pd.DataFrame(records))
    copy_input(Path(__file__), writer.temp / "trace_source.py")
    copy_input(
        WORK / "MEDIA_REVISION_EXTREME_TRACE_ADDENDUM.md",
        writer.temp / "AMENDMENT_ADDENDUM.md",
    )
    final = writer.finish(
        {
            "all_selected_extreme_volume_ratios_traced": True,
            "new_perturbations": False,
            "unchanged_estimates": True,
        }
    )
    assert not verify_run(final)
    (REV / "extreme_trace_package.json").write_text(
        json.dumps(
            {
                "run": str(final.relative_to(WORK)).replace("\\", "/"),
                "manifest_sha256": file_hash(final / "run_manifest.json"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Sealed volume-ratio trace:", final)
    print(
        pd.DataFrame(summaries)[
            [
                "target",
                "scope",
                "descriptor_id",
                "zero_alternate_denominator",
                "reference_cv_squared",
                "max_alternate_ratio",
                "zero_alternate_covariance_fraction",
            ]
        ].to_string(index=False)
    )
