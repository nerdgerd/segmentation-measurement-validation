"""Complete support, transport and coverage displays from saved results.

The study reports the two MRI cohorts only. Layout values are those of the submitted
supplement, so the four target/analysis rows are not stretched to the height the
earlier eight-row versions needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from types import SimpleNamespace
from matplotlib.ticker import FixedLocator, ScalarFormatter

from .labels import MODEL_NAMES

SOURCE, SYNTHETIC, POSITIVE, GREY = ("#536878", "#D45B3E", "#786799", "#626B72")
NEUTRAL = "#626B72"
MRI = ["brats_tcga", "upenn"]
NAMES = {"brats_tcga": "TCGA", "upenn": "UPENN"}
LAWS = ["additive", "biased_proportional", "heteroscedastic"]
MARKERS = ["o", "s", "^"]
keys = [(target, scope) for target in MRI for scope in ["full", "audit"]]
old = SimpleNamespace(LAW_NAMES=MODEL_NAMES)


def choose(data, target, scope):
    return data[(data.target == target) & (data.scope == scope)]


select = choose


def mechanism(context, targets, scopes, name):
    empirical = context.read("matching_calibration")
    population = context.read("matched_population_association")
    finite = context.read("matched_finite_summary")
    keys = [(t, s) for t in targets for s in scopes]
    nrows = len(keys)
    fig = plt.figure(figsize=(6.7, 5.2 if nrows == 2 else 6.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.05], hspace=0.64, wspace=0.35)
    axes = [fig.add_subplot(grid[0, j]) for j in range(2)]
    for i, (target, scope) in enumerate(keys):
        group = choose(empirical, target, scope).query(
            "battery == 'complete' and law == 'additive' and theta == 0"
        )
        for side, color, marker, offset in [
            ("reference", GREY, "x", -0.18),
            ("observed", SOURCE, "o", 0),
            ("synthetic", SYNTHETIC, "s", 0.18),
        ]:
            row = group[
                group.side == ("observed" if side == "reference" else side)
            ].iloc[0]
            columns = (
                ["reference_log_median", "reference_log_iqr"]
                if side == "reference"
                else ["disagreement_log_median", "disagreement_log_iqr"]
            )
            for ax, col in zip(axes, columns):
                ax.plot(row[col], i + offset, marker, c=color, ms=5)
    rows = [NAMES[t] + (" " + s if len(scopes) > 1 else "") for t, s in keys]
    for ax, title, xlabel in zip(
        axes,
        ["A  Empirical location", "Empirical spread"],
        ["Median of log relative moment", "IQR of log relative moment"],
    ):
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.set_yticks(range(nrows), rows)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[1].set_yticklabels([])
    axes[0].legend(
        handles=[
            Line2D([], [], marker=m, ls="", c=c, label=l)
            for m, c, l in [
                ("x", GREY, "Reference"),
                ("o", SOURCE, "Observed M"),
                ("s", SYNTHETIC, "Synthetic M"),
            ]
        ],
        loc="upper left",
        bbox_to_anchor=(-0.16, -0.3),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    left, right = (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
    for i, (target, scope) in enumerate(keys):
        for j, law in enumerate(LAWS):
            for theta, color, shift in [(0, SOURCE, -0.055), (0.9, POSITIVE, 0.055)]:
                y = i + (j - 1) * 0.24 + shift
                group = choose(population, target, scope).query("battery == 'complete'")
                r = group[(group.law == law) & (group.theta == theta)].iloc[0]
                left.plot(r.profile_spearman, y, MARKERS[j], c=color, ms=4)
                group = choose(finite, target, scope)
                r = group[(group.law == law) & (group.theta == theta)].iloc[0]
                right.plot([r.p025, r.p975], [y, y], c=color, lw=1)
                right.plot(r["median"], y, MARKERS[j], c=color, ms=3.5)
    for ax in [left, right]:
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.set_yticks(range(nrows), rows)
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
    left.set_title("B  Population association", loc="left")
    left.set_xlabel("CCC-profile Spearman association")
    right.set_title("C  Finite datasets", loc="left")
    right.set_xlabel("Median and central 95%")
    right.set_yticklabels(
        [f"n={int(choose(finite, t, s).patient_count.iloc[0])}" for t, s in keys],
        fontsize=8,
    )
    handles = [
        Line2D([], [], marker="o", ls="", c=c, label=f"θ = {theta:g}")
        for theta, c in [(0, SOURCE), (0.9, POSITIVE)]
    ]
    handles += [
        Line2D([], [], marker=m, ls="", c=GREY, label=l)
        for m, l in zip(MARKERS, ["Additive", "Bias/slope", "Heteroscedastic"])
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.025,
        0.016,
        "M: mean squared disagreement. All three marginal median/IQR targets matched.",
        fontsize=7.8,
    )
    fig.subplots_adjust(
        left=0.16, right=0.985, top=0.95, bottom=0.26 if nrows == 2 else 0.225
    )
    context.save(fig, name)


def coverage(context):
    d = context.read("coverage_prediction")
    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    colors = ["#536878", "#827496", "#333E47", "#B1A3C6"]
    for color, ((template, law), g) in zip(
        colors, d.groupby(["template", "law"], sort=True)
    ):
        for typ, marker in [("utility", "o"), ("contrast", "x")]:
            q = g[g.endpoint_type.eq(typ)]
            ax.scatter(
                q.predicted_coverage,
                q.coverage_all_datasets,
                c=color,
                marker=marker,
                s=28,
                label=f"{template} / {old.LAW_NAMES[law]}"
                if typ == "utility"
                else None,
            )
    ax.plot([0.38, 0.98], [0.38, 0.98], c=NEUTRAL, lw=0.8, label="Identity")
    ax.axhline(0.93, c=NEUTRAL, ls=":", lw=0.8, label="Original gate 0.930")
    ax.set_xlim(0.38, 0.98)
    ax.set_ylim(0.38, 0.98)
    ax.set_xlabel("Normal-centering predicted coverage")
    ax.set_ylabel("Observed all-dataset coverage")
    ax.set_title("Prediction leaves residual undercoverage", loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
    ax.text(
        0.02,
        0.96,
        "○ Fixed utilities   × Paired contrasts",
        transform=ax.transAxes,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.88)
    context.save(fig, "figureS3_coverage_diagnosis")


def transport(context):
    d = context.read("matched_transport_recovery")
    case = context.read("case_mix_variance_shift")
    fig, axs = plt.subplots(1, 2, figsize=(6.7, 3.6), gridspec_kw={"wspace": 0.5})
    groups = []
    names = []
    for (law, scenario), g in d.groupby(["law", "scenario"]):
        groups.append(g.mean_absolute_error.to_numpy())
        names.append(
            old.LAW_NAMES[law].replace("Bias + slope", "Bias/slope")
            + "\n"
            + ("changed" if scenario == "changed_conditional_law" else "unchanged")
        )
    axs[0].boxplot(
        groups,
        tick_labels=names,
        medianprops={"color": NEUTRAL},
        flierprops={"marker": ".", "markersize": 2},
    )
    axs[0].tick_params(axis="x", labelsize=6)
    axs[0].set_ylabel("Mean absolute population CCC error")
    axs[0].set_title("A  Matched-spread violations", loc="left")
    groups = [
        select(case, target, scope).log2_reference_variance_ratio
        for target, scope in keys
    ]
    direction = {"brats_tcga": "U → T", "upenn": "T → U"}
    labels = [direction[target] + " " + scope for target, scope in keys]
    axs[1].boxplot(
        groups,
        tick_labels=labels,
        vert=False,
        medianprops={"color": NEUTRAL},
        flierprops={"marker": ".", "markersize": 2},
    )
    axs[1].axvline(0, c=NEUTRAL, ls=":", lw=0.7)
    axs[1].set_xlabel("$\\log_2(v_{target}/v_{source})$")
    axs[1].set_title("B  Empirical directional shifts", loc="left")
    axs[1].tick_params(axis="y", labelsize=6)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.86, bottom=0.2)
    context.save(fig, "figureS2_transport_laws")


def complete_fidelity(display):
    profiles = display.read("profile_agreement")
    diagnostics = display.read("descriptor_disagreement")
    source, generated, original, light = ("#536878", "#D45B3E", "#30383D", "#C8CED2")
    targets = MRI
    rows = 2 * len(targets)
    directions = {"brats_tcga": "UPENN → TCGA", "upenn": "TCGA → UPENN"}
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.9, 3.7),
        gridspec_kw={"width_ratios": [0.8, 0.8, 1.25], "wspace": 0.4},
    )
    labels, support = ([], [])
    for index, target in enumerate(targets):
        for scope_index, scope in enumerate(["full", "audit"]):
            y = index * 2 + scope_index
            labels.append(
                directions[target]
                + "\n"
                + ("All patients" if scope == "full" else "Held-out")
            )
            p = profiles[(profiles.target == target) & (profiles.scope == scope)]
            d = diagnostics[
                (diagnostics.target == target) & (diagnostics.scope == scope)
            ]
            for method_index, (policy, color, marker) in enumerate(
                [("source_ccc", source, "o"), ("synthetic_ccc", generated, "^")]
            ):
                row = p[p.policy == policy].iloc[0]
                for ax, column in zip(
                    axes[:2], ["profile_spearman", "mean_absolute_ccc_error"]
                ):
                    ax.plot(
                        row[column],
                        y + (method_index - 0.5) * 0.24,
                        marker,
                        color=color,
                        mfc=color if scope == "full" else "white",
                        ms=4.5,
                    )
            finite = d.log2_rmse_ratio[np.isfinite(d.log2_rmse_ratio)].to_numpy()
            axes[2].boxplot(
                finite,
                positions=[y],
                orientation="horizontal",
                widths=0.42,
                whis=[5, 95],
                showfliers=False,
                patch_artist=True,
                boxprops={"fc": "#F8ECE7", "ec": generated},
                medianprops={"color": original},
            )
            axes[2].scatter(
                finite,
                y + 0.15 * np.sin(np.arange(len(finite)) * 2.399963229728653),
                s=4,
                alpha=0.4,
                color=original,
                edgecolors="none",
            )
            support.append(
                {
                    "target": target,
                    "scope": scope,
                    "feature_rows": len(d),
                    "finite": len(finite),
                }
            )
    for ax in axes:
        ax.set(ylim=(rows - 0.45, -0.55), yticks=range(rows))
        for y in range(1, len(targets)):
            ax.axhline(2 * y - 0.5, color=light, lw=0.5)
    axes[0].set(
        yticklabels=labels,
        xlim=(-0.05, 1.02),
        xticks=[0, 0.5, 1],
        xlabel="Spearman ρ with\nobserved CCC",
    )
    axes[1].set(
        yticklabels=[],
        xlim=(-0.02, profiles.mean_absolute_ccc_error.max() + 0.035),
        xlabel="MAE to\nobserved CCC",
    )
    axes[2].set(
        yticklabels=[],
        xlim=(
            diagnostics.log2_rmse_ratio.min() - 0.6,
            diagnostics.log2_rmse_ratio.max() + 2,
        ),
        xlabel="L = log₂ RMSE ratio",
    )
    axes[2].set_xscale("symlog", linthresh=2)
    axes[2].xaxis.set_major_locator(FixedLocator([-3, 0, 3, 16]))
    axes[2].xaxis.set_major_formatter(ScalarFormatter())
    axes[2].axvline(0, color=original, ls=":", lw=0.8)
    for ax, title in zip(
        axes, ["A  CCC correlation", "B  CCC error", "C  Magnitude of differences"]
    ):
        ax.set_title(title, loc="left", fontsize=9)
    fig.legend(
        handles=[
            Line2D([], [], marker=marker, color=color, ls="", label=label)
            for marker, color, label in [
                ("o", source, "Source CCC"),
                ("^", generated, "Generated CCC"),
            ]
        ],
        loc="upper center",
        bbox_to_anchor=(0.59, 1),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.24,
        0.02,
        "Filled: all patients. Open: held-out patients. All finite ratios retained.",
        fontsize=7.2,
    )
    fig.subplots_adjust(left=0.24, right=0.985, top=0.80, bottom=0.29)
    display.save(fig, "figureS_complete_fidelity")
    return support
