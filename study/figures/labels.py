"""Display wording for the immutable scientific table identifiers."""

import re
import matplotlib

# Display wording follows the revised manuscript text (8 November 2026): observed,
# generated and source CCC values; CCC thresholds; false retentions and exclusions;
# source-to-target directions; and the three simulation models named in Methods.
EXACT = {
    "A  Disagreement at θ = 0": "A  Mean squared differences, θ = 0",
    "B  CCC at θ = 0": "B  Corresponding CCC values",
    "Real log relative M": "Simulated observed differences\n(log relative M)",
    "Synthetic log relative M": "Simulated generated differences\n(log relative M)",
    "Real CCC": "Simulated observed CCC",
    "Synthetic CCC": "Simulated generated CCC",
    "log relative reference variance": "log relative between-patient variance",
    "C  Matched reference distributions and observed transfers": "C  Simulated correlation baselines",
    "CCC-profile Spearman association": "Spearman ρ, observed vs generated CCC",
    "θ = 0": "θ = 0, independent",
    "θ = 0.9": "θ = 0.9, associated",
    "Population": "Exact model value",
    "Finite median": "Median of simulated datasets",
    "Observed transfer": "Generated contours (all patients)",
    "TCGA target": "TCGA target cohort",
    "UPENN target": "UPENN target cohort",
    "Observed source": "Source CCC",
    "Synthetic target": "Generated CCC",
    "Observed target": "Observed CCC",
    "Candidate CCC": "Source or generated CCC",
    "Observed target CCC": "Observed CCC",
    "C  Disagreement magnitude: all finite descriptor values": "C  Magnitude of feature differences",
    "L = log₂ (synthetic RMSE / observed RMSE)": "L = log₂(generated RMSE / observed RMSE)",
    "A  Retained descriptors": "A  Features retained at each threshold",
    "Number retained": "Features retained",
    "CCC cutoff": "CCC threshold",
    "B  Rank clearance versus scale error": "B  Correlation versus magnitude of differences",
    "Rank clearance\nρ$_{synthetic}$ − matched no-coupling reference":
        "Spearman ρ minus simulated\ncorrelation baseline",
    "Calibration error: median |L|": "Median |L|",
    "ordering clears the null;\nmagnitude does not": "Correlation above\nsimulated baseline",
    "False retentions\nat τ = 0.95": "Falsely retained\nat 0.95",
    "C  Error tradeoff across all five cutoffs": "C  False retentions and false exclusions",
    "False-retention proportion": "False-retention rate",
    "False-exclusion proportion": "False-exclusion rate",
    "Complete decision-error counts": "False retentions / false exclusions",
    "more exclusions": "more false exclusions",
    "more retentions": "more false retentions",
    "FP − FN": "false retentions − false exclusions",
    "Overall disagreement rate": "Total disagreement",
    "Full": "All patients",
    "Audit": "Held-out patients",
    "Source": "Source CCC",
    "Synthetic": "Generated CCC",
    "S − Source": "Generated\n− source",
    "Source only  |  Both  |  Synthetic only": "Source only  |  Both  |  Generated only",
    "Selected-set score\n(mean real target CCC)": "Mean observed CCC\nof selected features",
    "Number of selected descriptors": "Features selected",
    "C  Real reliability of the differing selections": "C  Observed CCC of differing selections",
    "Real target CCC of the fixed selection categories": "Observed CCC of the fixed selections",
    "Synthetic only": "Generated only",
    "Synthetic measurement averaging on identical complete-K support":
        "Averaging feature values across generated contours",
    "CCC of mean of K draws − CCC of an individual draw": "CCC of the mean − CCC of individual contours",
    "Synthetic M": "Generated M",
    "Normal-centering predicted coverage": "Coverage predicted by centring approximation",
    "Observed all-dataset coverage": "Observed interval coverage",
    "Prediction leaves residual undercoverage": "Coverage remained below the required level",
    "Original gate 0.930": "Reporting requirement 0.930",
    "○ Fixed utilities   × Paired contrasts": "○ Fixed-set scores   × Score differences",
    "Mean absolute population CCC error": "Mean absolute model CCC error",
    "A  Matched-spread violations": "A  Transport under specified models",
    "B  Empirical directional shifts": "B  Reference-variance differences",
    "Additive": "Random",
    "Bias/slope": "Offset and proportional",
    "Heteroscedastic": "Measurement-dependent",
    "B  Population association": "B  Exact model values",
    "C  Finite datasets": "C  Simulated datasets",
    "M: mean squared disagreement. All three marginal median/IQR targets matched.":
        "M: mean squared feature difference. Marginal medians/IQRs matched.",
}
MODEL_NAMES = {"additive": "Random", "biased_proportional": "Offset and\nproportional",
               "heteroscedastic": "Measurement-\ndependent"}
DIRECTION_NAMES = [  # compact source-target forms first, then arrows between full cohort names
    (r"\bU ?→ ?T\b", "UPENN-to-TCGA"), (r"\bT ?→ ?U\b", "TCGA-to-UPENN"),
    (r" →\n", "-to-\n"), (r" → ", "-to-"),
]


def display_text(value):
    if value in EXACT:
        return EXACT[value]
    value = value.replace(" · Additive", " · Random").replace(" · Bias / slope", " · Offset and proportional")
    value = value.replace(" · Heteroscedastic", " · Measurement-dependent")
    value = re.sub(r"\bfull\b", "all", value)
    value = re.sub(r"\baudit\b", "held-out", value)
    value = re.sub(r"\bFull\b", "All patients", value)
    value = re.sub(r"\bAudit\b", "Held-out", value)
    value = value.replace("synthetic", "generated").replace("Synthetic", "Generated")
    value = value.replace("descriptors", "features")
    value = value.replace("Mean real target CCC", "Mean observed CCC")
    value = value.replace("Real split-half", "Observed split-half")
    value = value.replace(" · Syn.", " · Generated CCC").replace(" · Src.", " · Source CCC")
    value = value.replace("Filled: all. Open: held-out.", "Filled: all patients. Open: held-out patients.")
    value = re.sub(r"^observed ρ=(\S+)$", r"ρ = \1", value)
    value = re.sub(r"^Median ([+-]?[\d.]+)\s+\|L\| median ([\d.]+)$",
                   lambda m: f"median L = {m[1].replace('-', '−')}, median |L| = {m[2]}", value)
    if value.startswith("brats_tcga / "):  # coverage legend: one line per entry
        value = value.replace("brats_tcga / ", "TCGA · ")
        value = value.replace("-\n", "-").replace("\n", " ")
    for pattern, replacement in DIRECTION_NAMES:
        value = re.sub(pattern, replacement, value)
    return value


def relabel(fig):
    for text in fig.findobj(matplotlib.text.Text):
        text.set_text(display_text(text.get_text()))
    for ax in fig.axes:
        for getter, positions, setter in [
            (ax.get_xticklabels, ax.get_xticks, ax.set_xticks),
            (ax.get_yticklabels, ax.get_yticks, ax.set_yticks),
        ]:
            before = [text.get_text() for text in getter()]
            after = [display_text(value) for value in before]
            if before != after:
                setter(positions(), after)
