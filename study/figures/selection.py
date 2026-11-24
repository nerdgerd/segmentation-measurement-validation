"""Supplementary fixed-set selection and contour-averaging sensitivity."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .common import FigureContext

SOURCE, SYNTHETIC, REAL, LIGHT = ("#536878", "#D45B3E", "#30383D", "#C8CED2")
TARGETS = ["brats_tcga", "upenn"]
NAMES = {"brats_tcga": "TCGA", "upenn": "UPENN"}
DIRECTIONS = {"brats_tcga": "UPENN → TCGA", "upenn": "TCGA → UPENN"}
LAWS = ["additive", "biased_proportional", "heteroscedastic"]
LAW_NAMES = ["Additive", "Bias / slope", "Heteroscedastic"]
LAW_MARKERS = ["o", "s", "^"]


def choose(df, target, scope="full"):
    return df[(df.target == target) & (df.scope == scope)]


def close(a, b):
    if not np.allclose(a, b, atol=1e-12, rtol=1e-11, equal_nan=True):
        raise AssertionError(f"Stored values disagree: {a!r}, {b!r}")


def jitter(n, width=0.12):
    return width * np.sin(np.arange(n) * 2.399963229728653)


class SelectionFigures(FigureContext):
    def screening(self):
        anchored = self.read("anchored_utility")
        selections = self.read("selected_set_utility")
        overlap = self.read("selection_overlap")
        values = self.read("descriptor_prediction_errors")
        rows, category_rows = ([], [])
        for target in TARGETS:
            for scope in ["full", "audit"]:
                group = choose(anchored, target, scope)
                source = group[group.policy == "source_ccc"].iloc[0]
                synth = group[group.policy == "synthetic_ccc"].iloc[0]
                src_ids, syn_ids = (
                    set(json.loads(source.selected_ids)),
                    set(json.loads(synth.selected_ids)),
                )
                k = int(source.k)
                assert len(src_ids) == len(syn_ids) == k and source.k == synth.k
                for policy, ids in [
                    ("source_ccc", src_ids),
                    ("synthetic_ccc", syn_ids),
                ]:
                    saved = (
                        choose(selections, target, scope)
                        .query("policy == @policy")
                        .iloc[0]
                    )
                    assert (
                        ids == set(json.loads(saved.selected_ids))
                        and len(ids) == saved.budget
                    )
                categories = [
                    ("source_only", src_ids - syn_ids),
                    ("both", src_ids & syn_ids),
                    ("synthetic_only", syn_ids - src_ids),
                ]
                real = (
                    choose(values, target, scope)
                    .query("policy == 'source_ccc'")
                    .set_index("descriptor_id")
                )
                assert real.index.is_unique and src_ids.union(syn_ids).issubset(
                    set(real.index)
                )
                close(real.loc[sorted(src_ids), "observed_target_ccc"].mean(), source.U)
                close(real.loc[sorted(syn_ids), "observed_target_ccc"].mean(), synth.U)
                close(synth.U - source.U, source.synthetic_minus_source)
                ov = choose(overlap, target, scope)
                ov = ov[
                    (ov["first"] == "source_ccc") & (ov["second"] == "synthetic_ccc")
                    | (ov["second"] == "source_ccc") & (ov["first"] == "synthetic_ccc")
                ].iloc[0]
                close(len(src_ids & syn_ids) / k, ov.selected_set_overlap)
                record = {
                    "source": source.source,
                    "target": target,
                    "scope": scope,
                    "J": int(source.J),
                    "k": k,
                    "source_score": source.U,
                    "synthetic_score": synth.U,
                    "random_score": source.U_random,
                    "oracle_score": source.U_oracle,
                    "synthetic_minus_source": source.synthetic_minus_source,
                    "source_only": len(src_ids - syn_ids),
                    "both": len(src_ids & syn_ids),
                    "synthetic_only": len(syn_ids - src_ids),
                    "direct_lower": source.direct_lower,
                    "direct_upper": source.direct_upper,
                    "direct_undefined_fraction": source.direct_undefined_fraction,
                }
                rows.append(record)
                for category, ids in categories:
                    for descriptor in sorted(ids):
                        category_rows.append(
                            {
                                "source": source.source,
                                "target": target,
                                "scope": scope,
                                "category": category,
                                "descriptor_id": descriptor,
                                "real_target_ccc": real.loc[
                                    descriptor, "observed_target_ccc"
                                ],
                                "budget": k,
                                "construction_information": "saved_selection_IDs_only",
                                "evaluation_information": "real_target_CCC_attached_after_fixed_categories",
                            }
                        )
                close(
                    (
                        real.loc[sorted(syn_ids - src_ids), "observed_target_ccc"].sum()
                        - real.loc[
                            sorted(src_ids - syn_ids), "observed_target_ccc"
                        ].sum()
                    )
                    / k,
                    source.synthetic_minus_source,
                )
        scores, categories = (pd.DataFrame(rows), pd.DataFrame(category_rows))
        self.table(scores, "figure4_scores_and_overlap")
        self.table(categories, "figure4_category_points")
        fig = plt.figure(figsize=(6.9, 5.7))
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.35, 1],
            height_ratios=[1, 1.13],
            hspace=0.67,
            wspace=0.4,
        )
        a, b, c = (
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :]),
        )
        labels = []
        for i, row in scores.iterrows():
            label = (
                DIRECTIONS[row.target]
                + "\n"
                + ("Full" if row.scope == "full" else "Audit")
            )
            labels.append(label)
            a.plot(
                [row.random_score, row.oracle_score], [i, i], c=LIGHT, lw=1.4, zorder=0
            )
            a.plot(row.random_score, i, "|", c="#ADB5BA", ms=8)
            a.plot(row.oracle_score, i, "*", c="#ADB5BA", ms=6)
            a.plot([row.source_score, row.synthetic_score], [i, i], c=REAL, lw=1)
            for score, color, marker in [
                (row.source_score, SOURCE, "o"),
                (row.synthetic_score, SYNTHETIC, "^"),
            ]:
                a.plot(
                    score,
                    i,
                    marker,
                    c=color,
                    mfc=color if row.scope == "full" else "white",
                    ms=6,
                    mew=1.2,
                )
            status = "†" if not np.isfinite(row.direct_lower) else "‡"
            a.text(
                1.02,
                i,
                f"{row.synthetic_minus_source:+.3f}{status}",
                transform=a.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8.5,
            )
            start = 0
            for count, color in [
                (row.source_only, SOURCE),
                (row.both, LIGHT),
                (row.synthetic_only, SYNTHETIC),
            ]:
                b.barh(
                    i,
                    count,
                    left=start,
                    height=0.58,
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                )
                b.text(
                    start + count / 2,
                    i,
                    str(int(count)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if color in [SOURCE, SYNTHETIC] else REAL,
                )
                start += count
        for ax in [a, b]:
            ax.set(ylim=(3.55, -0.55), yticks=range(4))
            ax.axhline(1.5, color=LIGHT, lw=0.6)
        a.set(
            xlim=(0.63, 0.96),
            xticks=[0.65, 0.8, 0.95],
            yticklabels=labels,
            xlabel="Selected-set score\n(mean real target CCC)",
        )
        a.set_title("A  Selected sets and anchors", loc="left", fontsize=9.5, pad=24)
        a.text(
            1.02, 1.065, "S − Source", transform=a.transAxes, ha="left", fontsize=7.5
        )
        b.set(
            yticklabels=[],
            xlabel="Number of selected descriptors",
            xlim=(0, max(2 * scores.k - scores.both) + 1),
            xticks=[0, 24, 48, 72],
        )
        b.set_title("B  Fixed selection overlap", loc="left", fontsize=9.5, pad=24)
        b.text(
            0.5,
            1.035,
            "Source only  |  Both  |  Synthetic only",
            transform=b.transAxes,
            ha="center",
            fontsize=6.5,
        )
        cat_settings = [
            ("source_only", SOURCE, "o", -0.23),
            ("both", REAL, "s", 0),
            ("synthetic_only", SYNTHETIC, "^", 0.23),
        ]
        for i, row in scores.iterrows():
            g = choose(categories, row.target, row.scope)
            for cat, color, marker, offset in cat_settings:
                d = g[g.category == cat].sort_values("descriptor_id")
                v = d.real_target_ccc.to_numpy()
                c.scatter(
                    v,
                    i + offset + jitter(len(v), 0.065),
                    s=12,
                    marker=marker,
                    facecolors=color if row.scope == "full" else "none",
                    edgecolors=color,
                    linewidths=0.55,
                    alpha=0.55,
                )
                c.plot(np.mean(v), i + offset, "|", color=color, ms=10, mew=1.4)
        c.axhline(1.5, color=LIGHT, lw=0.6)
        c.set(
            ylim=(3.55, -0.55),
            yticks=range(4),
            yticklabels=labels,
            xlim=(-0.04, 1.02),
            xticks=[0, 0.25, 0.5, 0.75, 1],
            xlabel="Real target CCC of the fixed selection categories",
        )
        c.set_title(
            "C  Real reliability of the differing selections", loc="left", fontsize=9.5
        )
        c.legend(
            handles=[
                Line2D([], [], marker=marker, color=color, ls="", label=label)
                for label, color, marker, _ in [
                    ("Source only", SOURCE, "o", 0),
                    ("Both", REAL, "s", 0),
                    ("Synthetic only", SYNTHETIC, "^", 0),
                ]
            ]
            + [Line2D([], [], marker="|", color=REAL, ls="", label="Category mean")],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=4,
            frameon=False,
            fontsize=8,
        )
        fig.legend(
            handles=[
                Line2D([], [], marker=m, ls="", c=color, label=label)
                for m, color, label in [
                    ("o", SOURCE, "Source"),
                    ("^", SYNTHETIC, "Synthetic"),
                    ("|", "#ADB5BA", "Random"),
                    ("*", "#ADB5BA", "Sample oracle\n(evaluation only)"),
                ]
            ],
            loc="upper center",
            bbox_to_anchor=(0.57, 1.0),
            ncol=4,
            frameon=False,
            fontsize=8.5,
        )
        fig.text(
            0.18,
            0.015,
            "Filled: full. Open: audit.  † Withheld range.  ‡ Issued conditional range; uncalibrated.",
            fontsize=7.6,
        )
        fig.subplots_adjust(left=0.21, right=0.985, top=0.83, bottom=0.2)
        self.annotations["figure4"] = rows
        self.supports["figure4"] = {
            "separate_direction_scope_rows": 4,
            "fixed_categories_points": len(categories),
            "budgets": scores.k.tolist(),
            "selection_ids_reused": True,
            "ranges_displayed_as_CIs": False,
            "oracle": "evaluation-only sample anchor",
        }
        self.save(fig, "figure4_selection_utility")

    def averaging(self):
        values = self.read("current_aggregation_descriptor_values")
        summary = self.read("current_aggregation_summary")
        cohorts = [("brats_tcga", 200), ("upenn", 100)]
        final = []
        fig, ax = plt.subplots(figsize=(6.9, 2.5))
        records = []
        for i, (cohort, k) in enumerate(cohorts):
            d = values[(values.cohort == cohort) & (values.K == k)].sort_values(
                "descriptor_id"
            )
            assert len(d) == 192 and (d.final_K == k).all()
            close(d.draw_averaged - d.same_support_single, d.increase_same_support)
            final.append(d)
            finite = d.increase_same_support[
                np.isfinite(d.increase_same_support)
            ].to_numpy()
            row = summary[
                (summary.cohort == cohort)
                & (summary.K == k)
                & (summary.endpoint == "increase_same_support")
            ].iloc[0]
            close(np.median(finite), row["median"])
            close(np.quantile(finite, 0.95), row.p95)
            assert len(finite) == row.defined
            ax.boxplot(
                finite,
                positions=[i],
                orientation="horizontal",
                widths=0.4,
                whis=[5, 95],
                patch_artist=True,
                showfliers=False,
                boxprops={"fc": "#F8ECE7", "ec": SYNTHETIC},
                medianprops={"color": REAL},
            )
            ax.scatter(
                finite,
                i + jitter(len(finite), 0.17),
                s=9,
                c=SYNTHETIC,
                alpha=0.4,
                marker=["o", "s"][i],
                edgecolors="none",
            )
            records.append(
                {
                    "cohort": cohort,
                    "K": k,
                    "all_descriptors": 192,
                    "defined": len(finite),
                    "median": float(np.median(finite)),
                    "p95": float(np.quantile(finite, 0.95)),
                }
            )
            ax.text(
                0.98,
                0.85 - i * 0.6,
                f"n = {len(finite)} / 192; median {np.median(finite):.3f}",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )
        ax.axvline(0, c=REAL, ls=":", lw=1)
        ax.set(
            yticks=[0, 1],
            yticklabels=["TCGA · K = 200", "UPENN · K = 100"],
            ylim=(1.5, -0.5),
            xlabel="CCC of mean of K draws − CCC of an individual draw",
            xlim=(-0.08, 0.65),
        )
        ax.set_title(
            "Synthetic measurement averaging on identical complete-K support",
            loc="left",
        )
        fig.subplots_adjust(left=0.21, right=0.985, top=0.85, bottom=0.25)
        self.table(pd.concat(final, ignore_index=True), "figureS_averaging_points")
        self.annotations["averaging"] = records
        self.supports["averaging"] = {
            "all_descriptor_rows": 384,
            "defined_TCGA": 167,
            "defined_UPENN": 192,
            "TCGA_K": 200,
            "UPENN_K": 100,
            "target": "mean of measurements, not one contour",
        }
        self.save(fig, "figureS_averaging_descriptor_values")
