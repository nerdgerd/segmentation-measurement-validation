"""The current manuscript's experiments, ordered by scientific question."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    key: str
    question: str
    purpose: str
    module: str
    tables: tuple[str, ...]
    long_run: bool = False


EXPERIMENTS = (
    Experiment(
        "ranking-baseline",
        "Q1",
        "Exact model values for the simulated correlation baseline, permutations and sensitivities",
        "feature_ranking",
        (
            "matched_population_association",
            "matching_calibration",
            "permutation_summary",
        ),
        True,
    ),
    Experiment(
        "ranking-finite",
        "Q1",
        "Simulated datasets under the same models, patient counts and feature availability",
        "finite_ranking",
        ("matched_finite_summary",),
        True,
    ),
    Experiment(
        "measurement-differences",
        "Q2",
        "Replay feature RMSE ratios from recorded mean squared differences",
        "measurement_differences",
        ("descriptor_disagreement",),
    ),
    Experiment(
        "feature-selection",
        "Q3",
        "Feature selection at all five CCC thresholds with paired patient bootstraps",
        "feature_selection",
        (
            "experiment3_threshold_decision_metrics",
            "experiment3_threshold_decision_intervals",
        ),
        True,
    ),
    Experiment(
        "contour-averaging",
        "S3",
        "Individual generated contours versus means across generated contours",
        "contour_averaging",
        ("current_aggregation_summary",),
    ),
    Experiment(
        "coverage-and-selection",
        "S6/S8",
        "Top-quarter ranking anchors and the failed coverage check",
        "coverage_and_selection",
        ("anchored_utility", "coverage_diagnosis"),
    ),
    Experiment(
        "interval-definedness",
        "S8",
        "Reportable ranges and exclusive versus overlapping failure counts",
        "interval_definedness",
        ("bootstrap_definedness_summary",),
    ),
    Experiment(
        "bootstrap-reasons",
        "S8",
        "Guard reason for each stored bootstrap endpoint",
        "bootstrap_reasons",
        (),
    ),
    Experiment(
        "reference-variation",
        "S6",
        "Between-patient variance shifts by feature family and region",
        "reference_variation",
        ("case_mix_family_region_summary",),
    ),
    Experiment(
        "relative-scale",
        "diag.",
        "Zero means and continuous mean-relative-to-SD diagnostics",
        "relative_scale_diagnostics",
        (),
    ),
    Experiment(
        "extreme-measurements",
        "diag.",
        "Trace unusually large RMSE ratios to the saved feature values",
        "extreme_measurements",
        (),
    ),
    Experiment(
        "reciprocal-audit",
        "S5",
        "Reconstruction of both directions from individual generated contours",
        "reciprocal_audit",
        (),
    ),
)


def experiment(key):
    for item in EXPERIMENTS:
        if item.key == key:
            return item
    raise ValueError(f"Unknown experiment: {key}")
