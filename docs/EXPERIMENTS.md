# Experiments and code

The shared contour pipeline produces the inputs for all three experiments. Each
direction (UPENN-to-TCGA, TCGA-to-UPENN) is evaluated separately in the
all-patient and held-out analyses. Supplement sections are given as S1–S8.

## Shared contour pipeline

| Step | Entry point | Numerical implementation | Output |
|---|---|---|---|
| Prepare paired feature values, reference masks and source Dice values | `tools/stages.py prepare` | [prepare_inputs.py](../study/src/redesign2/prepare_inputs.py), [adapters/](../study/src/redesign2/adapters/) | Role-separated feature values, mask and Dice manifests |
| Generate contours at source Dice values | `tools/stages.py generate` | [generate.py](../study/src/redesign2/generate.py), [diffuse.py](../study/src/redesign2/adapters/diffuse.py) | Feature values of each generated contour, flags and calibration attempts |
| Fix eligible features and CCC values | `tools/stages.py construct` | [construction.py](../study/src/redesign2/construction.py) | Source and generated CCC values, moment caches, fixed selections and a commitment |
| Evaluate against observed CCC values and feature differences | `tools/stages.py evaluate` | [evaluate.py](../study/src/redesign2/evaluate.py) | `profile_agreement`, `descriptor_disagreement`, prediction errors and selection tables |

Construction sees the source contour pairs and the corrected target contours
only. Evaluation opens the original target contours after every direction and
analysis named in the locked protocol has been committed. Held-out CCC values
stay fixed in the training patients.

## Experiment 1 — correlation of CCC values (S4)

| Component | Run command | Code | Recorded results |
|---|---|---|---|
| Correlation of generated (and source) with observed CCC values | Shared `evaluate` step | `evaluate._profile_rows` | `profile_agreement.csv` |
| Exact model values for the simulated correlation baseline | `run ranking-baseline` | [feature_ranking.py](../study/experiments/feature_ranking.py), `mechanism.match_law`, `simulation.population_moments` | `matching_calibration.csv`, `matched_population_association.csv` |
| Simulated datasets with the cohorts' patient counts and feature availability | `run ranking-finite` | [finite_ranking.py](../study/experiments/finite_ranking.py), `simulation.hierarchical_draws`, `simulation_repetitions.PatientMoments` | `matched_finite_summary.csv` |
| Within-block permutation analysis | Included in `ranking-baseline` | `mechanism.permutation_indices`, `mechanism.permuted_associations` | `permutation_blocks.csv`, `permutation_summary.csv` |

The held-out simulations describe test-patient moments; the permutation analysis
recalculates CCC values in the test patients. Their correlations answer different
questions and are not interchangeable.

## Experiment 2 — magnitude of feature differences (S5)

The shared `evaluate` step calculates each feature's RMSE ratio from original and
generated feature values on matched support, through
`moments.measurement_difference_ratio`, `moments.compare_disagreement` and
`moments.summarize_disagreement` in [moments.py](../study/src/redesign2/moments.py).

`run measurement-differences` invokes
[measurement_differences.py](../study/experiments/measurement_differences.py).
It checks all 767 recorded finite ratios from the saved mean squared differences
and writes a separately labelled reference replay; it needs no patient data. The
original values remain in `descriptor_disagreement.csv`. Recalculation from
contours uses `evaluate`.

Principal outputs: `descriptor_disagreement.csv`,
`disagreement_family_summary.csv`, `descriptor_prediction_errors.csv`.

## Experiment 3 — feature selection (S7)

`run feature-selection` invokes
[feature_selection.py](../study/experiments/feature_selection.py), which calls
`threshold_experiment.run_threshold_experiment`. Selections at each threshold use
[threshold_decisions.py](../study/src/redesign2/threshold_decisions.py).

Features selected with source and with generated CCC values are compared with
those selected with observed CCC values at every threshold. Patient resampling is
paired across methods. In the all-patient analysis, CCC values are recalculated
for each resample; in the held-out analysis, the training-patient CCC values stay
fixed and the test patients are resampled. Undefined cells remain undefined, and
ranges are withheld where the reporting rule requires.

The six `experiment3_*` tables hold the point metrics, intervals, paired
differences, feature-level crossings, figure source and a ranking-sensitivity
check. `verify` reconstructs all 40 point cells from the supplied CCC values.

## Supplementary calculations

| Command | Purpose | Entry-point code | Outputs |
|---|---|---|---|
| `contour-averaging` | CCC values of individual generated contours versus means across contours (S3) | [contour_averaging.py](../study/experiments/contour_averaging.py) | `current_aggregation_summary.csv`, feature-level values |
| `ranking-baseline` | Moment-transport recovery, denominator sensitivity, precision-check stability and patient split halves (S4, S6, S8) | [feature_ranking.py](../study/experiments/feature_ranking.py) | `matched_transport_recovery`, `finite_pool_selection_stability`, `split_half_summary` |
| `coverage-and-selection` | Random and oracle anchors for the top-quarter ranking; the failed coverage check (S6, S8) | [coverage_and_selection.py](../study/experiments/coverage_and_selection.py) | `anchored_utility.csv`, `coverage_diagnosis.csv` |
| `interval-definedness` | Issued and withheld ranges, with exclusive and overlapping failure counts (S8) | [interval_definedness.py](../study/experiments/interval_definedness.py) | `bootstrap_definedness_summary.csv` |
| `bootstrap-reasons` | Reason code for every stored bootstrap endpoint (S8) | [bootstrap_reasons.py](../study/experiments/bootstrap_reasons.py) | Endpoint reason records |
| `reference-variation` | Between-patient variance differences by feature family and region (S6) | [reference_variation.py](../study/experiments/reference_variation.py) | `case_mix_family_region_summary.csv` |
| `relative-scale` | Zero means and mean-relative-to-SD checks (diagnostic; not reported in the supplement) | [relative_scale_diagnostics.py](../study/experiments/relative_scale_diagnostics.py) | Zero-point diagnostic summaries |
| `extreme-measurements` | Trace extreme RMSE ratios to the saved feature values (diagnostic; not reported in the supplement) | [extreme_measurements.py](../study/experiments/extreme_measurements.py) | Original and corrected feature-value traces |
| `reciprocal-audit` | Independent reconstruction of both directions, including the large TCGA-to-UPENN difference, from individual generated contours (S5) | [reciprocal_audit.py](../study/experiments/reciprocal_audit.py) | Audit record with 111 checks |

The dimensionless population, simulated-dataset and known-truth coverage
simulations are run by `tools/stages.py population`, `finite` and `coverage`.
Their numerical code is in `simulation.py`, `simulation_driver.py`,
`simulation_pipeline.py` and `simulation_repetitions.py`.

## Figures

`tools/reproduce.py figures` calls [build.py](../study/figures/build.py).

| Figures | Code |
|---|---|
| Figure 1, study overview: text labels only; vector graphics and numbers unchanged | [overview.py](../study/figures/overview.py) |
| Figures 2–4; complete feature-selection counts and agreement (Figures S6, S7) | [main.py](../study/figures/main.py) |
| Averaging (Figure S1) and top-quarter ranking (Figure S5) | [selection.py](../study/figures/selection.py) |
| Complete simulation results (Figure S2), CCC and RMSE comparisons (Figure S3), transport (Figure S4) and coverage (Figure S8) | [supplement.py](../study/figures/supplement.py) |
| Input paths, style and output provenance | [common.py](../study/figures/common.py) |

Every experiment module exposes `run(workspace, run_id, ...)`. Importing a module
reads no data and runs no computation. Stored table identifiers are mapped to the
manuscript wording for display only.
