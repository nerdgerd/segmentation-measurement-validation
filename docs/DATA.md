# Data and calculation rules

## Where data belong

`study/reference/` contains the feature-level result tables and plotting
coordinates. Imaging, patient-level tables, generated contour pools and
operational manifests are local inputs and are not distributed.
`tools/import_local_inputs.py` stages the recorded input bundle from the original
study checkout and records an explicit original-to-local path map.

The images are obtained under the original dataset terms. Contour identities,
cohort inclusion, region definitions and availability flags come from the locked
metadata. A missing identity or inclusion rule is an error; the code does not
infer a replacement from the current feature values.

## Weighting and support

Each patient contributes one corrected contour, one original contour and K_i
generated contours. For N patients, the original contour of patient i has weight
1/N and each of its generated contours 1/(N K_i), so every patient receives the
same total weight however many generated contours it has. Moments use
population-form weighted sums. Bootstrap resamples draw whole patients.

Each generated contour enters separately. Averaging feature values across
generated contours is a distinct sensitivity analysis. Explicit availability
flags determine support. A feature requires at least 10 eligible patients and
0.90 weighted support; the exact construction and evaluation rules are in the
locked specification. A missing CCC never becomes zero.

## Locked specifications

| Record | What it fixes |
|---|---|
| `study/config/protocol.lock.yaml` and `.sha256` | Cohorts, contour roles, generator, support, weighting, seed streams and construction/evaluation rules |
| `study/config/execution_resolution.yaml` | Execution choices fixed for contour generation |
| `study/config/simulation_resolution.yaml` | Original simulation execution resolution |
| `study/config/measurement_validation_amendment_v2.yaml` | Simulation models, independent and associated error scales, solver bounds, simulated datasets and diagnostics |
| `study/config/threshold_decision_fidelity_v1.yaml` | Five CCC thresholds, fixed feature sets, paired bootstraps and missingness rules |
| `study/config/media_revision_20260917.yaml` | Retained selection anchors and coverage diagnostics |
| `study/config/media_revision_extreme_trace_20260917.yaml` | Recorded rules for tracing extreme ratios |
| `study/config/environment.lock.json` and root `requirements.lock.txt` | Recorded environment and dependencies |

Recorded file names and seed namespaces are exact identifiers and are not renamed.

## Scope of the locked files

The locked specifications, the threshold specification
(`study/reference/tables/experiment3_threshold_analysis_spec.json`) and the dated
protocol notes in `provenance/protocol_notes/` are distributed unchanged, because
their hashes identify the analysis and its seeds. They were written for a
protocol that also included two exploratory CT cohorts (NSCLC-Interobserver1 and
LIDC-IDRI). The authors removed those cohorts from the study. Their rows were
removed from every supplied table and their number macros from
`study/reference/generated/numbers.tex`; every MRI record is unchanged byte for
byte. `provenance/CT_REMOVAL.json` lists each file's hash before and after,
with the rows removed.

The computational code still reads the full locked protocol, so the stages that
iterate over all protocol cohorts expect inputs for those cohorts as well.
Comparisons with the supplied tables are restricted to the two MRI cohorts.

## Column names

Machine identifiers keep their original meanings and are not renamed in the
hashed tables and configurations:

| Identifier | Meaning |
|---|---|
| `descriptor_id` | feature |
| `observed` | original contours (observed differences and CCC values) |
| `synthetic` | generated contours |
| `source_ccc`, `observed_source` | source CCC values, from the calibration cohort's contour pairs |
| `full`, `audit` | all-patient and held-out analyses |
| `source`, `target` | calibration and evaluation cohorts; `brats_tcga` is TCGA |
| `disagreement` | feature difference (M is its mean square) |

`evidence_table` and `evidence_row` identify the recorded source cell.

## Numerical precision

Some M columns were shortened by earlier CSV staging, while their saved L values
retain the original calculation. For example, an original M of
0.00010802472655189466 is stored as 0.0001080247265518. The Experiment 2
reference replay checks each ratio against the precision of those literal decimal
values and keeps the original L. The largest discrepancy is 6.324×10⁻¹³. Full
computation from patient moments uses the original formula; no result table is
rewritten to hide this serialisation difference.

## Provenance and output protection

Seeds derive from SHA-256 of the configured stream and explicit keys. A fresh run
name changes storage only. Scientific outputs are written atomically to new
directories with input hashes, specification hash, code hashes, environment,
seeds, weighting and estimand. Completed runs cannot be overwritten.

Original source locks validate their imported original bytes. New outputs
separately stamp the current code, including experiment and plotting modules.
Original code identities are never substituted for the current source hashes.
