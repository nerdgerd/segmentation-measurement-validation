"""Figures 2–4: ranking, measurement differences, and cutoff decisions."""

from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SOURCE = "#536878"
SYNTHETIC = "#D45B3E"
TARGET = "#30383D"
LIGHT = "#C8CED2"
TARGETS = ["brats_tcga", "upenn"]
NAMES = {"brats_tcga": "TCGA", "upenn": "UPENN"}
DIRECTIONS = {"brats_tcga": "UPENN → TCGA", "upenn": "TCGA → UPENN"}
LAWS = ["additive", "biased_proportional", "heteroscedastic"]
LAW_NAMES = ["Additive", "Bias / slope", "Heteroscedastic"]
LAW_MARKERS = ["o", "s", "^"]


def close(actual, expected, *, atol=1e-12, rtol=1e-11) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(f"stored values disagree: {actual!r} versus {expected!r}")


def choose(frame: pd.DataFrame, target: str, scope: str = "full") -> pd.DataFrame:
    return frame[(frame.target == target) & (frame.scope == scope)]


def build_figure2(context, outputs: dict[str, str]) -> None:
    points = context.read_file(context.data / "figure2_population_points.csv")
    finite = context.read_file(context.data / "figure2_finite_points.csv")
    summaries = context.read_file(context.data / "figure2_cell_summaries.csv")
    profiles = context.read_file(context.tables / "profile_agreement.csv")
    scatter = points.query("law == 'additive' and theta == 0 and scope == 'full'")
    assert len(scatter) == 384 and len(finite) == 24000
    fig = plt.figure(figsize=(6.9, 5.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.67, wspace=0.45)
    panel_a = fig.add_subplot(grid[0, 0])
    panel_b = fig.add_subplot(grid[0, 1])
    vmin = scatter.reference_log_relative_variance.min()
    vmax = scatter.reference_log_relative_variance.max()
    cmap = matplotlib.colormaps["Greys"].resampled(256)
    color_points = None
    for index, target in enumerate(TARGETS):
        subset = choose(scatter, target)
        marker = ("o", "s")[index]
        panel_a.scatter(
            subset.real_log_relative_M,
            subset.synthetic_log_relative_M,
            s=10,
            marker=marker,
            facecolors=TARGET if index == 0 else "none",
            edgecolors=TARGET,
            linewidths=0.55,
            alpha=0.5,
        )
        color_points = panel_b.scatter(
            subset.real_ccc,
            subset.synthetic_ccc,
            c=subset.reference_log_relative_variance,
            cmap=cmap,
            vmin=vmin - 1,
            vmax=vmax + 1,
            s=13,
            marker=marker,
            alpha=0.8,
            edgecolors=TARGET,
            linewidths=0.25,
        )
        rho_m = spearmanr(
            subset.real_log_relative_M, subset.synthetic_log_relative_M
        ).statistic
        rho_c = spearmanr(subset.real_ccc, subset.synthetic_ccc).statistic
        panel_a.text(
            0.04,
            0.95 - index * 0.13,
            f"{NAMES[target]}: ρ = {rho_m:+.3f}",
            transform=panel_a.transAxes,
            va="top",
            fontsize=8,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1},
        )
        panel_b.text(
            0.04,
            0.95 - index * 0.13,
            f"{NAMES[target]}: ρ = {rho_c:.3f}",
            transform=panel_b.transAxes,
            va="top",
            fontsize=8,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1},
        )
    panel_a.set(xlabel="Real log relative M", ylabel="Synthetic log relative M")
    panel_a.set_title("A  Disagreement at θ = 0", loc="left")
    panel_b.plot([0, 1], [0, 1], ":", color=LIGHT, lw=1, zorder=0)
    panel_b.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xlabel="Real CCC",
        ylabel="Synthetic CCC",
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
    )
    panel_b.set_title("B  CCC at θ = 0", loc="left")
    cbar = fig.colorbar(color_points, ax=panel_b, fraction=0.055, pad=0.035)
    cbar.set_label("log relative reference variance", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", ls="", color=TARGET, mfc=TARGET, label="TCGA target"
            ),
            Line2D(
                [],
                [],
                marker="s",
                ls="",
                color=TARGET,
                mfc="none",
                label="UPENN target",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.51, 1.005),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    panel_c = fig.add_subplot(grid[1, :])
    labels = []
    for target_index, target in enumerate(TARGETS):
        for law_index, law in enumerate(LAWS):
            y = target_index * 3 + law_index
            labels.append(f"{NAMES[target]} · {LAW_NAMES[law_index]}")
            for theta, color, offset in ((0, SOURCE, -0.13), (0.9, SYNTHETIC, 0.13)):
                subset = choose(finite, target).query("law == @law and theta == @theta")
                values = subset.sample_profile_spearman.to_numpy()
                stored = (
                    choose(summaries, target)
                    .query("law == @law and theta == @theta")
                    .iloc[0]
                )
                assert len(values) == 2000
                close(np.median(values), stored.finite_median)
                close(
                    np.quantile(values, [0.025, 0.975]),
                    [stored.finite_p025, stored.finite_p975],
                )
                violin = panel_c.violinplot(
                    values,
                    positions=[y + offset],
                    vert=False,
                    widths=0.22,
                    showextrema=False,
                    points=150,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.35)
                panel_c.plot(np.median(values), y + offset, "|", color=color, ms=8)
                panel_c.plot(
                    stored.population_rho,
                    y + offset,
                    LAW_MARKERS[law_index],
                    color=color,
                    markeredgecolor=TARGET,
                    markeredgewidth=0.5,
                    ms=5,
                    zorder=4,
                )
        empirical = (
            choose(profiles, target)
            .query("policy == 'synthetic_ccc'")
            .iloc[0]
            .profile_spearman
        )
        start = target_index * 3 - 0.34
        end = target_index * 3 + 2.34
        panel_c.vlines(empirical, start, end, color=TARGET, ls=":", lw=1.2, zorder=5)
        panel_c.plot(
            empirical,
            target_index * 3 + 1,
            "D",
            color=TARGET,
            mfc="white",
            ms=5.5,
            zorder=6,
        )
        panel_c.annotate(
            f"observed ρ={empirical:.3f}",
            (empirical, target_index * 3 + 1),
            xytext=(4, -10 if target_index == 0 else 8),
            textcoords="offset points",
            fontsize=6.5,
            color=TARGET,
        )
    panel_c.axhline(2.5, color=LIGHT, lw=0.7)
    panel_c.set(
        ylim=(5.5, -0.5),
        xlim=(0, 1.02),
        yticks=range(6),
        yticklabels=labels,
        xlabel="CCC-profile Spearman association",
        xticks=[0, 0.25, 0.5, 0.75, 1],
    )
    panel_c.set_title(
        "C  Matched reference distributions and observed transfers", loc="left", pad=10
    )
    panel_c.legend(
        handles=[
            Patch(fc=SOURCE, alpha=0.5, label="θ = 0"),
            Patch(fc=SYNTHETIC, alpha=0.5, label="θ = 0.9"),
            Line2D([], [], marker="o", ls="", color=TARGET, label="Population"),
            Line2D([], [], marker="|", ls="", color=TARGET, label="Finite median"),
            Line2D(
                [],
                [],
                marker="D",
                ls=":",
                color=TARGET,
                mfc="white",
                label="Observed transfer",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
        frameon=False,
        fontsize=7.2,
    )
    fig.subplots_adjust(left=0.13, right=0.92, top=0.91, bottom=0.15)
    position = panel_c.get_position()
    panel_c.set_position([0.28, position.y0, 0.64, position.height])
    panel_c.tick_params(axis="y", labelsize=7.5)
    context.save(fig, "figure2_mechanism", outputs)


def _jitter(count: int, width: float = 0.12) -> np.ndarray:
    return width * np.sin(np.arange(count) * 2.399963229728653)


def build_figure3(context, outputs: dict[str, str]) -> None:
    points = context.read_file(context.data / "figure3_ccc_points.csv")
    ratios = context.read_file(context.data / "figure3_rmse_points.csv")
    profiles = context.read_file(context.tables / "profile_agreement.csv")
    assert len(points) == 768 and len(ratios) == 384
    all_values = np.r_[points.construction_score, points.observed_target_ccc]
    limits = (min(-0.04, all_values.min() - 0.02), max(1.02, all_values.max() + 0.02))
    fig = plt.figure(figsize=(6.9, 4.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.68], hspace=0.48, wspace=0.26)
    for index, target in enumerate(TARGETS):
        ax = fig.add_subplot(grid[0, index])
        ax.plot(limits, limits, ":", color=LIGHT, lw=1, zorder=0)
        annotations = []
        for policy, color, marker, label in (
            ("source_ccc", SOURCE, "o", "Source"),
            ("synthetic_ccc", SYNTHETIC, "^", "Synthetic"),
        ):
            subset = choose(points, target).query("policy == @policy")
            assert len(subset) == 192 and (not subset.descriptor_id.duplicated().any())
            ax.scatter(
                subset.observed_target_ccc,
                subset.construction_score,
                s=13,
                alpha=0.42,
                marker=marker,
                c=color,
                edgecolors="none",
            )
            rho = spearmanr(
                subset.construction_score, subset.observed_target_ccc
            ).statistic
            mae = np.mean(
                np.abs(subset.construction_score - subset.observed_target_ccc)
            )
            stored = choose(profiles, target).query("policy == @policy").iloc[0]
            close(rho, stored.profile_spearman)
            close(mae, stored.mean_absolute_ccc_error)
            annotations.append(f"{label}: ρ={rho:.3f}, MAE={mae:.3f}")
        ax.text(
            0.04,
            0.95,
            "\n".join(annotations),
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            bbox={"fc": "white", "ec": "none", "alpha": 0.82, "pad": 2},
        )
        ax.set(
            xlim=limits,
            ylim=limits,
            xticks=[0, 0.5, 1],
            yticks=[0, 0.5, 1],
            xlabel="Observed target CCC",
            ylabel="Candidate CCC" if index == 0 else "",
        )
        ax.set_title(f"{'AB'[index]}  {DIRECTIONS[target]}", loc="left")
    panel_c = fig.add_subplot(grid[1, :])
    for index, target in enumerate(TARGETS):
        subset = choose(ratios, target).sort_values("descriptor_id")
        values = subset.log2_rmse_ratio.to_numpy()
        assert len(values) == 192 and np.isfinite(values).all()
        panel_c.boxplot(
            values,
            positions=[index],
            orientation="horizontal",
            widths=0.38,
            whis=[5, 95],
            showfliers=False,
            patch_artist=True,
            boxprops={"fc": "#F8ECE7", "ec": SYNTHETIC, "lw": 0.9},
            medianprops={"color": TARGET, "lw": 1.3},
            whiskerprops={"color": SYNTHETIC},
            capprops={"color": SYNTHETIC},
        )
        panel_c.scatter(
            values,
            index + _jitter(len(values), 0.19),
            s=8,
            c=TARGET,
            alpha=0.42,
            marker=("o", "s")[index],
            edgecolors="none",
            zorder=3,
        )
        panel_c.text(
            0.99,
            0.83 - index * 0.51,
            f"Median {np.median(values):+.3f}   |L| median {np.median(np.abs(values)):.3f}",
            transform=panel_c.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 1},
        )
    panel_c.axvline(0, color=TARGET, ls=":", lw=1)
    panel_c.set_xscale("symlog", linthresh=2, linscale=1)
    panel_c.set(
        xlim=(ratios.log2_rmse_ratio.min() - 0.6, ratios.log2_rmse_ratio.max() + 2),
        ylim=(1.5, -0.5),
        yticks=[0, 1],
        yticklabels=[DIRECTIONS[target].replace(" → ", " →\n") for target in TARGETS],
        xlabel="L = log₂ (synthetic RMSE / observed RMSE)",
    )
    panel_c.xaxis.set_major_locator(FixedLocator([-3, -1, 0, 1, 3, 8, 16]))
    panel_c.xaxis.set_major_formatter(ScalarFormatter())
    panel_c.set_title(
        "C  Disagreement magnitude: all finite descriptor values", loc="left"
    )
    panel_c.text(
        0.5,
        -0.57,
        "−1: half the observed RMSE     0: equal     +1: twice the observed RMSE",
        transform=panel_c.transAxes,
        ha="center",
        fontsize=8,
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=SOURCE, label="Observed source"),
            Line2D(
                [], [], marker="^", ls="", color=SYNTHETIC, label="Synthetic target"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.9, bottom=0.17)
    context.save(fig, "figure3_measurement_transfer", outputs)


def rank_scale_rows(context) -> pd.DataFrame:
    profiles = context.read_file(context.tables / "profile_agreement.csv")
    population = context.read_file(
        context.tables / "matched_population_association.csv"
    )
    disagreement = context.read_file(context.tables / "descriptor_disagreement.csv")
    thresholds = context.read_file(
        context.tables / "experiment3_threshold_decision_metrics.csv"
    )
    rows = []
    for target in TARGETS:
        for scope in ("full", "audit"):
            profile = (
                choose(profiles, target, scope)
                .query("policy == 'synthetic_ccc'")
                .iloc[0]
            )
            null_rows = choose(population, target, scope).query(
                "battery == 'complete' and theta == 0"
            )
            assert set(null_rows.law) == set(LAWS)
            null_reference = float(null_rows.profile_spearman.median())
            rmse = choose(disagreement, target, scope).query(
                "policy == 'synthetic_ccc'"
            )
            median_abs = float(np.nanmedian(np.abs(rmse.log2_rmse_ratio)))
            direction = (
                "upenn_to_brats_tcga"
                if target == "brats_tcga"
                else "brats_tcga_to_upenn"
            )
            threshold = thresholds[
                (thresholds.direction == direction)
                & (thresholds.analysis_scope == scope)
                & (thresholds.method == "synthetic_target")
                & np.isclose(thresholds.threshold, 0.95)
            ].iloc[0]
            rows.append(
                {
                    "direction": direction,
                    "direction_label": DIRECTIONS[target],
                    "target": target,
                    "scope": scope,
                    "synthetic_profile_rho": float(profile.profile_spearman),
                    "matched_no_coupling_reference": null_reference,
                    "rank_clearance": float(profile.profile_spearman - null_reference),
                    "median_absolute_log2_rmse_ratio": median_abs,
                    "false_retentions_at_0_95": int(threshold.fp),
                    "false_exclusions_at_0_95": int(threshold.fn),
                    "null_summary": "median of the three prespecified matched theta=0 population laws",
                }
            )
    result = pd.DataFrame(rows)
    expected_fp = [18, 21, 92, 93]
    assert result.false_retentions_at_0_95.tolist() == expected_fp
    result.to_csv(
        context.data / "figure4_rank_scale_points.csv", index=False, lineterminator="\n"
    )
    return result


def build_figure4(context, outputs: dict[str, str]) -> None:
    data = context.read_file(context.data / "figure4_threshold_decision_source.csv")
    rank_scale = rank_scale_rows(context)
    directions = ["upenn_to_brats_tcga", "brats_tcga_to_upenn"]
    titles = ["UPENN → TCGA", "TCGA → UPENN"]
    styles = {"full": ("-", "o"), "audit": ("--", "s")}
    colors = {"synthetic_target": SYNTHETIC, "observed_source": SOURCE}
    base = data[data.metric == "jaccard"].copy()
    fig = plt.figure(figsize=(7.25, 6.35))
    outer = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 1], width_ratios=[1.02, 1], hspace=0.4, wspace=0.48
    )
    count_grid = outer[0, 0].subgridspec(2, 1, hspace=0.13)
    count_axes = [fig.add_subplot(count_grid[index, 0]) for index in range(2)]
    for index, (direction, title) in enumerate(zip(directions, titles)):
        ax = count_axes[index]
        for scope, (line, marker) in styles.items():
            cell = base[(base.direction == direction) & (base.analysis_scope == scope)]
            target_rows = cell[cell.method == "synthetic_target"]
            ax.plot(
                target_rows.threshold,
                target_rows.n_target_retained,
                line,
                color=TARGET,
                marker=marker,
                ms=3.8,
                lw=1.0,
                mfc="white" if scope == "audit" else TARGET,
            )
            for method, color in colors.items():
                subset = cell[cell.method == method]
                ax.plot(
                    subset.threshold,
                    subset.n_method_retained,
                    line,
                    color=color,
                    marker=marker,
                    ms=3.8,
                    lw=1.0,
                    mfc="white" if scope == "audit" else color,
                )
        ax.text(0.01, 0.91, title, transform=ax.transAxes, fontsize=8.3, va="top")
        ax.set_ylim(0, 150)
        ax.set_yticks([0, 50, 100, 150])
        ax.set_xticks([0.75, 0.8, 0.85, 0.9, 0.95])
        ax.grid(axis="y", color=LIGHT, lw=0.6, alpha=0.65)
        if index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("CCC cutoff", fontsize=8.5)
    count_axes[0].set_title("A  Retained descriptors", loc="left", fontsize=10.5, pad=7)
    fig.text(0.018, 0.705, "Number retained", rotation=90, va="center", fontsize=8.5)
    rank_ax = fig.add_subplot(outer[0, 1])
    xlim = (-0.19, 0.54)
    ylim = (-0.05, 1.58)
    rank_ax.axvspan(
        0,
        xlim[1],
        ymin=(0 - ylim[0]) / (ylim[1] - ylim[0]),
        color=SYNTHETIC,
        alpha=0.07,
        zorder=0,
    )
    rank_ax.axvline(0, color=TARGET, lw=0.8, ls=":")
    rank_ax.axhline(0, color=TARGET, lw=0.8, ls=":")
    direction_colors = {"upenn_to_brats_tcga": SOURCE, "brats_tcga_to_upenn": SYNTHETIC}
    offsets = {
        ("upenn_to_brats_tcga", "full"): (6, 5),
        ("upenn_to_brats_tcga", "audit"): (6, -13),
        ("brats_tcga_to_upenn", "full"): (8, -7),
        ("brats_tcga_to_upenn", "audit"): (-72, 8),
    }
    for row in rank_scale.itertuples():
        color = direction_colors[row.direction]
        marker = "o" if row.scope == "full" else "s"
        size = 28 + 2.1 * row.false_retentions_at_0_95
        rank_ax.scatter(
            row.rank_clearance,
            row.median_absolute_log2_rmse_ratio,
            s=size,
            marker=marker,
            c=color if row.scope == "full" else "white",
            edgecolors=color,
            linewidths=1.2,
            zorder=3,
        )
        label = (
            "U→T" if row.direction == "upenn_to_brats_tcga" else "T→U"
        ) + f" {row.scope}"
        rank_ax.annotate(
            label,
            (row.rank_clearance, row.median_absolute_log2_rmse_ratio),
            xytext=offsets[row.direction, row.scope],
            textcoords="offset points",
            fontsize=6.8,
            color=TARGET,
        )
    rank_ax.text(
        0.98,
        0.62,
        "ordering clears the null;\nmagnitude does not",
        transform=rank_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=SYNTHETIC,
    )
    rank_ax.set(
        xlim=xlim,
        ylim=ylim,
        xlabel="Rank clearance\nρ$_{synthetic}$ − matched no-coupling reference",
        ylabel="Calibration error: median |L|",
    )
    rank_ax.grid(color=LIGHT, lw=0.55, alpha=0.55)
    rank_ax.set_title(
        "B  Rank clearance versus scale error", loc="left", fontsize=10.5, pad=7
    )
    size_handles = [
        plt.scatter(
            [],
            [],
            s=28 + 2.1 * count,
            facecolors="white",
            edgecolors=TARGET,
            label=str(count),
        )
        for count in (20, 50, 90)
    ]
    rank_ax.legend(
        handles=size_handles,
        title="False retentions\nat τ = 0.95",
        loc="lower right",
        frameon=False,
        fontsize=6.5,
        title_fontsize=6.5,
        borderpad=0.2,
        labelspacing=0.6,
    )
    trade_grid = outer[1, :].subgridspec(1, 2, wspace=0.24)
    trade_axes = [fig.add_subplot(trade_grid[0, index]) for index in range(2)]
    for index, (direction, title) in enumerate(zip(directions, titles)):
        ax = trade_axes[index]
        for scope, (line, marker) in styles.items():
            for method, color in colors.items():
                xrows = data[
                    (data.direction == direction)
                    & (data.analysis_scope == scope)
                    & (data.method == method)
                    & (data.metric == "false_retention_proportion")
                ].sort_values("threshold")
                yrows = data[
                    (data.direction == direction)
                    & (data.analysis_scope == scope)
                    & (data.method == method)
                    & (data.metric == "false_exclusion_proportion")
                ].sort_values("threshold")
                assert np.array_equal(
                    xrows.threshold.to_numpy(), yrows.threshold.to_numpy()
                )
                x = xrows.point_estimate.to_numpy()
                y = yrows.point_estimate.to_numpy()
                ax.plot(
                    x,
                    y,
                    line,
                    color=color,
                    marker=marker,
                    ms=4.3,
                    lw=1.1,
                    mfc="white" if scope == "audit" else color,
                    mec=color,
                    zorder=3,
                )
                reported = (
                    xrows.interval_status.eq("reported").to_numpy()
                    & yrows.interval_status.eq("reported").to_numpy()
                )
                if reported.any():
                    xerr = np.vstack(
                        [
                            x[reported] - xrows.bootstrap_p025.to_numpy()[reported],
                            xrows.bootstrap_p975.to_numpy()[reported] - x[reported],
                        ]
                    )
                    yerr = np.vstack(
                        [
                            y[reported] - yrows.bootstrap_p025.to_numpy()[reported],
                            yrows.bootstrap_p975.to_numpy()[reported] - y[reported],
                        ]
                    )
                    ax.errorbar(
                        x[reported],
                        y[reported],
                        xerr=xerr,
                        yerr=yerr,
                        fmt="none",
                        ecolor=color,
                        elinewidth=0.65,
                        capsize=1.4,
                        alpha=0.45,
                        zorder=2,
                    )
                withheld = ~reported
                if withheld.any():
                    ax.scatter(
                        x[withheld],
                        y[withheld],
                        marker="x",
                        s=15,
                        c=TARGET,
                        linewidths=0.55,
                        zorder=4,
                    )
                if scope == "full":
                    ax.annotate(
                        ".75",
                        (x[0], y[0]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        color=color,
                        fontsize=6.3,
                    )
                    ax.annotate(
                        ".95",
                        (x[-1], y[-1]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        color=color,
                        fontsize=6.3,
                    )
        ax.set_title(title, fontsize=9.2, pad=5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(color=LIGHT, lw=0.55, alpha=0.65)
        ax.set_xlabel("False-retention proportion", fontsize=8.5)
        if index == 0:
            ax.set_ylabel("False-exclusion proportion", fontsize=8.5)
    trade_axes[0].text(
        0,
        1.17,
        "C  Error tradeoff across all five cutoffs",
        transform=trade_axes[0].transAxes,
        fontsize=10.5,
    )
    method_handles = [
        Line2D([], [], color=SYNTHETIC, marker="o", label="Synthetic target"),
        Line2D([], [], color=SOURCE, marker="o", label="Observed source"),
        Line2D([], [], color=TARGET, marker="o", label="Observed target"),
    ]
    scope_handles = [
        Line2D([], [], color=TARGET, ls="-", marker="o", mfc=TARGET, label="Full"),
        Line2D([], [], color=TARGET, ls="--", marker="s", mfc="white", label="Audit"),
        Line2D([], [], color=TARGET, ls="", marker="x", label="Range withheld"),
    ]
    fig.legend(
        handles=method_handles + scope_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=6,
        frameon=False,
        fontsize=7.1,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.text(
        0.5,
        0.006,
        "Panel C: endpoint labels are .75 and .95; intermediate points are .80, .85 and .90. Error bars are conditional bootstrap ranges where reportable; × marks a withheld range.",
        ha="center",
        fontsize=6.4,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.925, bottom=0.105)
    context.save(fig, "figure4_threshold_decision_fidelity_continuous", outputs)


def build_supplement_threshold_figures(context, outputs: dict[str, str]) -> None:
    data = context.read_file(context.data / "figure4_threshold_decision_source.csv")
    directions = ["upenn_to_brats_tcga", "brats_tcga_to_upenn"]
    base = data[data.metric == "jaccard"].copy()
    rows = []
    labels = []
    direction_short = {"upenn_to_brats_tcga": "U→T", "brats_tcga_to_upenn": "T→U"}
    method_short = {"synthetic_target": "Syn.", "observed_source": "Src."}
    for direction in directions:
        for scope in ("full", "audit"):
            for method in ("synthetic_target", "observed_source"):
                subset = base[
                    (base.direction == direction)
                    & (base.analysis_scope == scope)
                    & (base.method == method)
                ].sort_values("threshold")
                rows.append(subset)
                labels.append(
                    f"{direction_short[direction]} · {scope.title()} · {method_short[method]}"
                )
    imbalance = np.array(
        [
            [row.fp.iloc[index] - row.fn.iloc[index] for index in range(5)]
            for row in rows
        ]
    )
    bound = max(1, float(np.abs(imbalance).max()))
    cmap = LinearSegmentedColormap.from_list(
        "decision_imbalance", [SOURCE, LIGHT, SYNTHETIC]
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    image = ax.imshow(
        imbalance,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    )
    for row_index, row in enumerate(rows):
        for column_index, item in enumerate(row.itertuples()):
            value = imbalance[row_index, column_index]
            color = "white" if abs(value) > 0.58 * bound else TARGET
            ax.text(
                column_index,
                row_index,
                f"{int(item.fp)}/{int(item.fn)}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    ax.set_xticks(range(5), [".75", ".80", ".85", ".90", ".95"])
    ax.set_yticks(range(8), labels, fontsize=7.3)
    ax.set_xlabel("CCC cutoff", fontsize=8)
    ax.set_title("Complete decision-error counts", loc="left", fontsize=10.5, pad=7)
    for boundary in (1.5, 3.5, 5.5):
        ax.axhline(boundary, color="white", lw=1.8)
    bar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.1,
        pad=0.24,
        ticks=[-bound, 0, bound],
    )
    bar.ax.set_xticklabels(
        ["more exclusions", "balanced", "more retentions"], fontsize=7
    )
    bar.set_label("FP − FN", fontsize=7.5, labelpad=1)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.9, bottom=0.29)
    context.save(fig, "figureS7_threshold_decision_counts", outputs)
    intervals = context.read_file(
        context.tables / "experiment3_threshold_decision_intervals.csv"
    )
    styles = {"full": ("-", "o"), "audit": ("--", "s")}
    colors = {"synthetic_target": SYNTHETIC, "observed_source": SOURCE}
    metrics = [
        ("jaccard", "Jaccard agreement"),
        ("decision_disagreement_rate", "Overall disagreement rate"),
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 4.6),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0.2, "wspace": 0.16},
    )
    for column, (direction, title) in enumerate(
        zip(directions, ["UPENN → TCGA", "TCGA → UPENN"])
    ):
        axes[0, column].set_title(title, fontsize=10, pad=7)
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            for scope, (line, marker) in styles.items():
                for method, color in colors.items():
                    subset = intervals[
                        (intervals.direction == direction)
                        & (intervals.analysis_scope == scope)
                        & (intervals.method == method)
                        & (intervals.metric == metric)
                    ].sort_values("threshold")
                    x = subset.threshold.to_numpy()
                    y = subset.point_estimate.to_numpy()
                    ax.plot(
                        x,
                        y,
                        line,
                        color=color,
                        marker=marker,
                        ms=4.5,
                        lw=1.15,
                        mfc="white" if scope == "audit" else color,
                        mec=color,
                        zorder=3,
                    )
                    reported = subset.interval_status.eq("reported").to_numpy()
                    if reported.any():
                        low = y[reported] - subset.bootstrap_p025.to_numpy()[reported]
                        high = subset.bootstrap_p975.to_numpy()[reported] - y[reported]
                        ax.errorbar(
                            x[reported],
                            y[reported],
                            yerr=np.vstack([low, high]),
                            fmt="none",
                            ecolor=color,
                            elinewidth=0.7,
                            capsize=1.7,
                            alpha=0.55,
                            zorder=2,
                        )
                    if (~reported).any():
                        ax.scatter(
                            x[~reported],
                            y[~reported],
                            marker="x",
                            s=17,
                            c=TARGET,
                            linewidths=0.6,
                            zorder=4,
                        )
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.grid(axis="y", color=LIGHT, lw=0.6, alpha=0.65)
            if column == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
            if row == 1:
                ax.set_xlabel("CCC cutoff")
                ax.set_xticks([0.75, 0.8, 0.85, 0.9, 0.95])
    fig.legend(
        handles=[
            Line2D([], [], color=SYNTHETIC, marker="o", label="Synthetic target"),
            Line2D([], [], color=SOURCE, marker="o", label="Observed source"),
            Line2D([], [], color=TARGET, ls="-", marker="o", label="Full"),
            Line2D(
                [], [], color=TARGET, ls="--", marker="s", mfc="white", label="Audit"
            ),
            Line2D([], [], color=TARGET, ls="", marker="x", label="Range withheld"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        fontsize=7.6,
    )
    fig.subplots_adjust(left=0.1, right=0.985, top=0.86, bottom=0.11)
    context.save(fig, "figureS7_threshold_agreement", outputs)
