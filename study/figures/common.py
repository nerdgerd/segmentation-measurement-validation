"""One explicit input/output boundary for all manuscript figures."""

from pathlib import Path
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .labels import relabel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style(section):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5 if section == "complete" else 8,
            "ytick.labelsize": {"main": 8, "selection": 8.5, "complete": 9}[section],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "medical-physics-readability-v1",
        }
    )


def _panel_heading(fig, renderer, x, y, letter, title):
    mark = fig.text(x, y, letter, fontsize=13, fontweight="bold", va="baseline")
    width = mark.get_window_extent(renderer).width / fig.bbox.width
    fig.text(x + width + .012, y, title, fontsize=10.5, fontweight="bold", va="baseline")


def finish_figure2(fig):
    """The longer key labels for panel C need two rows instead of one."""
    (panel_c,) = [ax for ax in fig.axes if ax.get_legend() is not None]
    old = panel_c.get_legend()
    handles, labels = old.legend_handles, [text.get_text() for text in old.get_texts()]
    assert labels == ["θ = 0, independent", "θ = 0.9, associated", "Exact model value",
                      "Median of simulated datasets", "Generated contours (all patients)"]
    old.remove()
    order = [0, 2, 1, 3, 4]  # column-major with ncol=3: θ in the first column
    panel_c.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order],
                   loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3, frameon=False, fontsize=7.2,
                   columnspacing=1.4)


def _one_minus_log(values):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(np.clip(1 - np.asarray(values, float), 1e-12, None))


def _one_minus_log_inverse(values):
    return 1 - 10 ** (-np.asarray(values, float))


CCC_TICKS = [0, .5, .75, .9, .95, .99, .999]


def finish_figure3(fig):
    """Panels A/B on a logarithmic 1 - CCC scale; the stored CCC values are not changed."""
    panel_a, panel_b, _ = fig.axes
    low, high = -.03, .9995
    for ax in (panel_a, panel_b):
        values = np.vstack([collection.get_offsets() for collection in ax.collections])
        assert len(values) == 384 and values.min() > low and values.max() < high, "a CCC would be clipped"
        (equality,) = [line for line in ax.lines if line.get_linestyle() == ":"]
        equality.set_data([low, high], [low, high])
        ax.set_xscale("function", functions=(_one_minus_log, _one_minus_log_inverse))
        ax.set_yscale("function", functions=(_one_minus_log, _one_minus_log_inverse))
        ax.set(xlim=(low, high), ylim=(low, high))
        labels = [f"{tick:g}" for tick in CCC_TICKS]
        ax.set_xticks(CCC_TICKS, labels)
        ax.set_yticks(CCC_TICKS, labels)
        ax.minorticks_off()
        ax.tick_params(labelsize=7)
        ax.grid(color="#C8CED2", lw=.5, alpha=.55, zorder=0)
        ax.set_axisbelow(True)
    # On this scale the highest values reach the top-left corner, so the statistics move
    # above each panel, under its title, where they cannot cover a point.
    for ax in (panel_a, panel_b):
        (box,) = ax.texts
        box.set(position=(0, 1.02), va="bottom", ha="left", bbox=None, fontsize=7.5)
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=27)
    fig.set_size_inches(6.9, 5.3)
    fig.subplots_adjust(top=.85)
    renderer = fig.canvas.get_renderer()
    for ax in (panel_a, panel_b):
        assert ax.texts[0].get_window_extent(renderer).y0 >= ax.get_window_extent(renderer).y1


def finish_figure4(fig):
    """Figure 4 layout after relabelling: the panel-C note moves to the caption,
    headings become bold letter + title, panel-B points carry direction names,
    and the key sits below every panel with each entry naming the panels it
    applies to. Plotted artists are not touched."""
    count_top, count_bottom, rank_ax, trade_left, trade_right = fig.axes
    assert count_top.get_title(loc="left") == "A  Features retained at each threshold"
    assert rank_ax.get_title(loc="left") == "B  Correlation versus magnitude of differences"
    (heading_c,) = [text for text in trade_left.texts if text.get_text().startswith("C  ")]
    (note,) = [text for text in fig.texts if text.get_text().startswith("Panel C:")]
    (ylabel,) = [text for text in fig.texts if text.get_text() == "Features retained"]
    (key,) = fig.legends
    point_labels = [text for text in rank_ax.texts if text.get_text().split(" ")[0] in
                    {"UPENN-to-TCGA", "TCGA-to-UPENN"}]
    assert len(point_labels) == 4
    for artist in (note, heading_c, key, *point_labels):
        artist.remove()
    count_top.set_title("", loc="left")
    rank_ax.set_title("", loc="left")

    # Filled/open markers already encode all-patient versus held-out (shared key), so
    # one label per direction replaces the four abbreviated point labels.
    for direction, color, align, shift in [("UPENN-to-TCGA", "#536878", "left", -6),
                                           ("TCGA-to-UPENN", "#D45B3E", "right", 10)]:
        points = np.array([label.xy for label in point_labels if label.get_text().startswith(direction)])
        anchor = (points[:, 0].min() if align == "left" else points[:, 0].max(), points[:, 1].min())
        rank_ax.annotate(direction, anchor, xytext=(shift, -15), textcoords="offset points",
                         ha=align, va="top", fontsize=7.4, color=color, fontweight="bold")
    # Panel-A direction names move to the empty lower-left corner, clear of the lines.
    for ax in (count_top, count_bottom):
        (name,) = ax.texts
        name.set(position=(.015, .06), va="bottom")

    # A wider row gap keeps the C heading clear of the panel-A threshold label.
    fig.set_size_inches(7.25, 6.75)
    count_top.get_subplotspec().get_topmost_subplotspec().get_gridspec().hspace = .62
    fig.subplots_adjust(left=.105, right=.985, top=.95, bottom=.16)
    ylabel.set_y((count_bottom.get_position().y0 + count_top.get_position().y1) / 2)

    renderer = fig.canvas.get_renderer()
    row = count_top.get_position().y1 + .018
    _panel_heading(fig, renderer, .006, row, "A", "Features retained at each threshold")
    _panel_heading(fig, renderer, rank_ax.get_tightbbox(renderer).x0 / fig.bbox.width, row, "B",
                   "Correlation versus magnitude of differences")
    _panel_heading(fig, renderer, .006, trade_left.title.get_window_extent(renderer).y1 / fig.bbox.height + .022,
                   "C", "False retentions and false exclusions")

    observed = "#30383D"
    entries = [  # column-major with ncol=3: row 1 colours, row 2 patient scope
        ("Generated CCC (A, C)", {"color": "#D45B3E", "marker": "o"}),
        ("All patients (A–C)", {"color": observed, "marker": "o", "mfc": observed}),
        ("Source CCC (A, C)", {"color": "#536878", "marker": "o"}),
        ("Held-out patients (A–C)", {"color": observed, "ls": "--", "marker": "s", "mfc": "white"}),
        ("Observed CCC (A)", {"color": observed, "marker": "o"}),
        ("Range withheld (C)", {"color": observed, "ls": "", "marker": "x"}),
    ]
    key_top = min(ax.get_tightbbox(renderer).y0 for ax in (trade_left, trade_right)) / fig.bbox.height - .025
    fig.legend(handles=[Line2D([], [], label=label, **style) for label, style in entries],
               loc="upper center", bbox_to_anchor=(.5, key_top), ncol=3, frameon=True, fancybox=False,
               edgecolor="#C8CED2", framealpha=1, fontsize=7.6, columnspacing=1.6, handletextpad=.5, borderpad=.6)


def finish_coverage(fig):
    """The relabelled model names no longer fit an in-axes legend; the key moves below."""
    (ax,) = fig.axes
    old = ax.get_legend()
    handles, labels = old.legend_handles, [text.get_text() for text in old.get_texts()]
    old.remove()
    fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(.5, 0), ncol=3,
               frameon=False, fontsize=7, columnspacing=1.6)
    fig.subplots_adjust(bottom=.3)


def finish_transport(fig):
    """Direction names replace the U/T/N/L abbreviations; panel B needs the room."""
    fig.subplots_adjust(wspace=.75)


FINISH = {
    "figure2_mechanism": finish_figure2,
    "figure3_measurement_transfer": finish_figure3,
    "figure4_threshold_decision_fidelity_continuous": finish_figure4,
    "figureS3_coverage_diagnosis": finish_coverage,
    "figureS2_transport_laws": finish_transport,
}


class FigureContext:
    """Plot stored tables; record every read and every exported file."""

    def __init__(self, reference, output):
        self.reference = Path(reference).resolve()
        self.p = self.project = Path(output).resolve()
        self.tables = self.reference / "tables"
        self.data = self.project / "figure_data"
        self.figs = self.project / "figs"
        self.inputs, self.outputs, self.annotations, self.supports = {}, {}, {}, {}
        for folder in (self.data, self.figs):
            folder.mkdir(parents=True, exist_ok=True)
        # Main panels use supplied coordinates; supplemental panels save theirs.
        for path in (self.reference / "figure_data").glob("*.csv"):
            destination = self.data / path.name
            destination.write_bytes(path.read_bytes())
            self.inputs["figure_data/" + path.name] = sha(path)

    def read(self, name):
        path = self.tables / (name + ".csv")
        return self.read_file(path)

    def read_file(self, path):
        path = Path(path)
        relative = (
            path.relative_to(self.reference)
            if path.is_relative_to(self.reference)
            else path.relative_to(self.project)
        )
        self.inputs[relative.as_posix()] = sha(path)
        return pd.read_csv(path)

    def table(self, frame, name):
        path = self.data / (name + ".csv")
        frame.to_csv(path, index=False, lineterminator="\n")
        self.outputs[path.relative_to(self.project).as_posix()] = sha(path)

    def save(self, fig, name, outputs=None):
        relabel(fig)
        if name == "figureS7_threshold_decision_counts":
            fig.subplots_adjust(left=0.40)
        elif name in FINISH:
            FINISH[name](fig)
        for suffix in ("pdf", "svg", "png"):
            path = self.figs / (name + "." + suffix)
            metadata = (
                {"CreationDate": None, "ModDate": None}
                if suffix == "pdf"
                else {"Date": None}
                if suffix == "svg"
                else None
            )
            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
                **({"metadata": metadata} if metadata is not None else {}),
            )
            self.outputs[path.relative_to(self.project).as_posix()] = sha(path)
        plt.close(fig)
