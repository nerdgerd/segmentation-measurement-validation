"""Simulate finite datasets with the matched laws and recorded patient support."""

from __future__ import annotations
from redesign2.execution import read_text
from pathlib import Path
from redesign2.execution import copy_input
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, time
import numpy as np
import pandas as pd
from redesign2.provenance import read_protocol, RunWriter
from redesign2.simulation import PopulationLaw, hierarchical_draws
from redesign2.simulation_pipeline import build_template
from redesign2.simulation_repetitions import SimulationSupport, PatientMoments
from redesign2.mechanism import association, distribution
from redesign2.execution import read_csv, storage_run_id
from types import SimpleNamespace


def worker(spec):
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    cfg, identity, template, support, laws, start, stop = spec
    real, syn = laws
    u = len(template.patients)
    j = len(template.descriptor_ids)
    real_units = np.repeat(np.arange(u), template.alternate_counts)
    syn_units = np.repeat(np.arange(u), 100)
    rp = PatientMoments(
        template,
        real_units,
        support.real_available,
        support.reference_available,
        min_patients=10,
        min_coverage=0.9,
    )
    sp = PatientMoments(
        template,
        syn_units,
        support.synthetic_available.reshape(-1, j),
        support.reference_available,
        min_patients=10,
        min_coverage=0.9,
    )
    rows = []
    local = dict(cfg)
    local["randomness"] = dict(cfg["randomness"])
    local["randomness"]["streams"] = dict(cfg["randomness"]["streams"])
    local["randomness"]["streams"]["finite"] = (
        cfg["randomness"]["streams"]["finite"]
        + "__"
        + "__".join(map(str, identity.values()))
    )
    for n in range(start, stop):
        data = hierarchical_draws(
            local, template, real, syn, stream="finite", replicate=n, draw_count=100
        )
        r = rp.pool(rp.integrate(data["reference"], data["real"]))
        s = sp.pool(sp.integrate(data["reference"], data["synthetic"].reshape(-1, j)))
        rr, ss = (r["ccc"][0], s["ccc"][0])
        finite = np.isfinite(rr) & np.isfinite(ss)
        rows.append(
            dict(
                identity,
                replicate=n,
                patient_count=len(set(template.patients)),
                unit_count=len(template.patients),
                observed_contours=sum(template.alternate_counts),
                draw_count=100,
                descriptors=j,
                evaluable_descriptors=int(finite.sum()),
                sample_profile_spearman=association(rr, ss),
                complete_battery=bool(finite.all()),
                real_undefined_fraction=float(np.mean(~np.isfinite(rr))),
                synthetic_undefined_fraction=float(np.mean(~np.isfinite(ss))),
                template_sha256=template.identity,
                availability_sha256=support.identity,
            )
        )
    return (identity, start, pd.DataFrame(rows))


def run(workspace, run_id, *, workers=1, pilot=False):
    """Simulate finite datasets with the matched laws and recorded patient support."""
    W = Path(workspace).resolve()
    args = SimpleNamespace(workers=workers, pilot=pilot)
    cfg = read_protocol(W / "config/measurement_validation_amendment_v2.yaml")
    basecfg = read_protocol(W / "config/protocol.lock.yaml")
    P = W / "critical_revision"
    analysis = W / json.loads(read_text(P / "analysis_package.json"))["run"]
    params = read_csv(analysis / "matched_law_parameters.csv")
    O = W / "runs" / cfg["revision"]["inherited_protocol_sha256"]
    inputs = [
        Path(__file__),
        P / "AMENDMENT_LOCK.json",
        W / "config/measurement_validation_amendment_v2.yaml",
        analysis / "run_manifest.json",
        analysis / "matched_law_parameters.csv",
    ]
    jobs = []
    blocked = []
    template_cache = {}
    for key, g in params.groupby(
        ["source", "target", "scope", "law", "theta"], sort=True
    ):
        source, target, scope, law, theta = key
        identity = dict(zip(["source", "target", "scope", "law", "theta"], key))
        laws = []
        side_ids = []
        for side in ["observed", "synthetic"]:
            q = g[g.side.eq(side)]
            side_ids.append(tuple(q.descriptor_id))
            laws.append(
                PopulationLaw(
                    **{
                        name: q[name].to_numpy()
                        for name in [
                            "source_mean",
                            "source_variance",
                            "bias",
                            "slope",
                            "residual_sd",
                        ]
                    },
                    heteroscedastic_log_sd_slope=float(
                        q.heteroscedastic_log_sd_slope.iloc[0]
                    ),
                )
                if len(q)
                else None
            )
        if not all((l is not None for l in laws)):
            blocked.append(dict(identity, reason="one_side_unmatched"))
            continue
        assert side_ids[0] == side_ids[1]
        ids = side_ids[0]
        generation = next(O.glob(f"generate-*-to-{target}-100"))
        if target not in template_cache:
            template_cache[target] = build_template(
                W, basecfg, O / "prepared-20260915", target, generation_run=generation
            )
        template, support, paths, _ = template_cache[target]
        positions = np.array([template.descriptor_ids.index(d) for d in ids])
        unit_mask = np.ones(len(template.patients), bool)
        if scope == "audit":
            split = read_csv(
                W / basecfg["resolved_metadata"]["patient_split"], dtype=str
            )
            patients = set(
                split.loc[
                    split.cohort.eq(target) & split.assignment.eq("audit"), "patient_id"
                ]
            )
            unit_mask = np.array([p in patients for p in template.patients])
        real_mask = np.repeat(unit_mask, template.alternate_counts)
        template = replace(
            template,
            descriptor_ids=ids,
            families=tuple((template.families[j] for j in positions)),
            patients=tuple((p for p, k in zip(template.patients, unit_mask) if k)),
            nodules=tuple((p for p, k in zip(template.nodules, unit_mask) if k)),
            alternate_counts=tuple(
                (p for p, k in zip(template.alternate_counts, unit_mask) if k)
            ),
        )
        support = SimulationSupport(
            support.reference_available[unit_mask][:, positions],
            support.real_available[real_mask][:, positions],
            support.synthetic_available[unit_mask][:, :, positions],
        )
        inputs.extend(paths)
        for start in range(0, 2000, 100):
            jobs.append(
                (cfg, identity, template, support, tuple(laws), start, start + 100)
            )
    if args.pilot:
        pilot = next(
            (s for s in jobs if s[1]["target"] == "upenn" and s[1]["scope"] == "full")
        )
        pilot = (*pilot[:-2], 0, 10)
        before = time.perf_counter()
        worker(pilot)
        elapsed = time.perf_counter() - before
        rec = {
            "pilot_datasets": 10,
            "seconds": elapsed,
            "cells": len(jobs) // 20,
            "total_datasets": len(jobs) * 100,
            "workers": args.workers,
            "ideal_compute_minutes_at_pilot_rate": elapsed
            / 10
            * len(jobs)
            * 100
            / args.workers
            / 60,
            "outputs_inspected_for_performance": False,
        }
        (P / "FINITE_COST_PILOT.json").write_text(
            json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(rec, indent=2))
        return
    writer = RunWriter(
        W,
        cfg,
        "matched_finite_mechanism",
        storage_run_id(run_id, "matched-finite-v1"),
        inputs=list(dict.fromkeys(inputs)),
        role="post_analysis_descriptive_actual_patient_support",
        estimand="single_random_contour_finite_CCC_profile_association",
        seeds={
            "master_seed": 2026091702,
            "namespace": "measurement_validation_amendment_v2",
            "stream": "finite",
            "cell_stream_suffix": "source__target__scope__law__theta",
            "patient_field_key": [
                "patient_fields",
                "template_identity",
                "absolute_replicate",
                "field_kind",
            ],
        },
    )
    frames = []
    before = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            identity, start, frame = f.result()
            frames.append(frame)
            folder = writer.temp / "batches"
            folder.mkdir(exist_ok=True)
            name = "__".join(map(str, identity.values())) + f"__{start:04d}.csv"
            frame.to_csv(folder / name, index=False)
            if n % 12 == 0 or n == len(futures):
                print(
                    f"{n}/{len(futures)} batches; elapsed {(time.perf_counter() - before) / 60:.1f} min",
                    flush=True,
                )
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["source", "target", "scope", "law", "theta", "replicate"]
    )
    summaries = []
    for key, g in result.groupby(["source", "target", "scope", "law", "theta"]):
        summaries.append(
            dict(
                zip(["source", "target", "scope", "law", "theta"], key),
                **distribution(g.sample_profile_spearman),
                patient_count=int(g.patient_count.iloc[0]),
                unit_count=int(g.unit_count.iloc[0]),
                observed_contours=int(g.observed_contours.iloc[0]),
                draw_count=100,
                descriptors=int(g.descriptors.iloc[0]),
                complete_dataset_fraction=float(g.complete_battery.mean()),
                evaluable_min=int(g.evaluable_descriptors.min()),
                evaluable_median=float(g.evaluable_descriptors.median()),
                evaluable_max=int(g.evaluable_descriptors.max()),
                mean_real_undefined_fraction=float(g.real_undefined_fraction.mean()),
                mean_synthetic_undefined_fraction=float(
                    g.synthetic_undefined_fraction.mean()
                ),
                template_sha256=g.template_sha256.iloc[0],
                availability_sha256=g.availability_sha256.iloc[0],
            )
        )
    writer.write_table("matched_finite_replicates.csv", result)
    writer.write_table("matched_finite_summary.csv", pd.DataFrame(summaries))
    writer.write_table(
        "unmatched_cells.csv",
        pd.DataFrame(
            blocked, columns=["source", "target", "scope", "law", "theta", "reason"]
        ),
    )
    copy_input(__file__, writer.temp / "finite_source.py")
    final = writer.finish(
        {"workers": args.workers, "elapsed_seconds": time.perf_counter() - before}
    )
    (P / "finite_package.json").write_text(
        json.dumps({"run": str(final.relative_to(W)).replace("\\", "/")}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Completed", final)
