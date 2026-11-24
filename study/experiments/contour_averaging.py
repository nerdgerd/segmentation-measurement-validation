"""Compare one-contour CCC with CCC after averaging contour measurements."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from redesign2.provenance import RunWriter, read_protocol
from redesign2.construction import reference_matrix, synthetic_matrix
from redesign2.empirical_math import aggregate_patient_moments
from redesign2.moments import weighted_joint_moments
from redesign2.mechanism import distribution
from redesign2.execution import read_csv, storage_run_id


def run(workspace, run_id):
    """Compare one-contour CCC with CCC after averaging contour measurements."""
    W = Path(workspace).resolve()
    cfg = read_protocol(W / "config/measurement_validation_amendment_v2.yaml")
    O = W / "runs" / cfg["revision"]["inherited_protocol_sha256"]
    P = W / "critical_revision"
    inputs = [
        Path(__file__),
        P / "AMENDMENT_LOCK.json",
        W / "config/measurement_validation_amendment_v2.yaml",
    ]
    rows = []
    for source, target, draws in [
        ("upenn", "brats_tcga", 200),
        ("brats_tcga", "upenn", 100),
    ]:
        root = O / f"generate-{source}-to-{target}-{draws}"
        construction = O / f"construct-{source}-to-{target}-full-{draws}"
        refpath = O / f"prepared-20260915/{target}/reference_features.csv"
        scorepath = construction / "synthetic_moments.csv"
        scores = read_csv(scorepath)
        ids = tuple(scores.descriptor_id)
        inputs.extend(
            [
                refpath,
                scorepath,
                root / "draw_features.csv",
                root / "draw_definedness.csv",
            ]
        )
        ref = read_csv(
            refpath, dtype={"patient_id": str, "unit_id": str, "annotation_id": str}
        )
        ref = ref[ref.descriptor_id.isin(ids)]
        values = read_csv(
            root / "draw_features.csv", dtype={"patient_id": str, "unit_id": str}
        )
        flags = read_csv(
            root / "draw_definedness.csv", dtype={"patient_id": str, "unit_id": str}
        )
        rm = reference_matrix(ref, cohort=target, descriptor_ids=ids)
        sm = synthetic_matrix(rm, values, flags, cohort=target)
        primary = aggregate_patient_moments(
            sm.patient_moments(), min_independent_patients=1, min_weighted_support=0.9
        )
        for j, d in enumerate(ids):
            assert np.isclose(
                primary.moments[j].ccc,
                scores.set_index("descriptor_id").loc[d, "ccc"],
                equal_nan=True,
                atol=1e-10,
                rtol=1e-10,
            )
        u = len(rm.keys)
        cube = sm.alternate.reshape(u, draws, len(ids))
        available = sm.eligible.reshape(u, draws, len(ids))
        for k in sorted({1, 20, 100, draws}):
            for j, d in enumerate(ids):
                support = available[:, :k, j]
                complete = support.all(axis=1)
                coverage = float(complete.mean())
                n = int(complete.sum())
                reason = (
                    ""
                    if n >= 10 and coverage >= 0.9
                    else "insufficient_complete_K_patient_support"
                )
                mean_ccc = np.nan
                same_ccc = np.nan
                if not reason:
                    x = rm.reference[complete, j]
                    y = cube[complete, :k, j]
                    mean_ccc = weighted_joint_moments(
                        x, y.mean(axis=1), np.ones(n) / n
                    ).ccc
                    same_ccc = weighted_joint_moments(
                        np.repeat(x, k), y.ravel(), np.ones(n * k) / (n * k)
                    ).ccc
                current = primary.moments[j].ccc
                rows.append(
                    dict(
                        source=source,
                        target=target,
                        cohort=target,
                        K=k,
                        final_K=draws,
                        descriptor_id=d,
                        current_primary_single=current,
                        same_support_single=same_ccc,
                        draw_averaged=mean_ccc,
                        increase_same_support=mean_ccc - same_ccc,
                        increase_vs_current_primary=mean_ccc - current,
                        complete_K_patients=n,
                        complete_K_support=coverage,
                        support_matches_primary=bool(
                            complete.all() and available[:, :, j].all()
                        ),
                        reason=reason,
                        primary_CCC_exactly_verified=True,
                    )
                )
    frame = pd.DataFrame(rows)
    summaries = []
    for key, g in frame.groupby(["cohort", "K", "final_K"]):
        for endpoint in ["increase_same_support", "increase_vs_current_primary"]:
            summaries.append(
                dict(
                    zip(["cohort", "K", "final_K"], key),
                    endpoint=endpoint,
                    **distribution(g[endpoint]),
                    p95=float(g[endpoint].quantile(0.95)),
                    all_descriptors=len(g),
                    support_changes=int((~g.support_matches_primary).sum()),
                )
            )
    writer = RunWriter(
        W,
        cfg,
        "current_pool_aggregation",
        storage_run_id(run_id, "current-aggregation-v1"),
        inputs=inputs,
        role="current_MRI_battery_and_actual_Diffuse_donor_pool",
        estimand="single_random_draw_versus_mean_of_K_current_draws_with_explicit_support",
    )
    writer.write_table("current_aggregation_descriptor_values.csv", frame)
    writer.write_table("current_aggregation_summary.csv", pd.DataFrame(summaries))
    final = writer.finish()
    (P / "aggregation_package.json").write_text(
        json.dumps({"run": str(final.relative_to(W)).replace("\\", "/")}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(final)
