# Validation

## What the current checkout checks

| Check | Evidence |
|---|---|
| Locked inputs | The configurations, the threshold specification, Figure 1's source and the plotting inputs match the publication baseline byte for byte |
| Result tables | Every MRI record of every supplied table is byte-identical to the baseline; the CT records were removed (`provenance/CT_REMOVAL.json`) |
| Numerical code | Syntax checks confirm that every numerical expression in the 31 core modules is unchanged, including an expansion of the extracted RMSE-ratio helper |
| Feature selection | All 40 direction/analysis/method/threshold cells reconstruct from the saved feature CCC values |
| Magnitude of feature differences | All 767 finite RMSE ratios reconstruct within the precision of the saved M columns; zero and undefined strata are preserved |
| Figures | Rebuilt in the recorded environment, all 12 figures and the 10 plotted-data files are byte-identical to those of the submitted manuscript; `provenance/CT_REMOVAL.json` lists their hashes |
| Figure 1 | Vector paths and numerical annotations are identical to the preserved original PDF |
| Module imports | Tests forbid data reads and run creation while every experiment module is imported |
| Input relocation | Tests check explicit path mapping and keep original locked code hashes separate from current source hashes |
| Command interface | The principal results print from an unrelated directory without patient inputs |

The numerical syntax check ignores documentation, the declared reference-path
rename, verified path resolution and source-provenance enumeration. It expands
the extracted ratio helper back into the original comparison before checking it.
It does not accept a newly tuned formula.

## Computational replay

Two saved runs of the full 2,000-bootstrap feature-selection experiment match the
MRI rows of all six supplied tables across 152 column comparisons, allowing only
the storage name to differ. The independent raw-artifact audit of both
directions passes 111 checks below the patient-moment caches. Hosted
checks run the tests, the identity verification, the summary, the Experiment 2
replay and the complete figure pipeline in a fresh Linux checkout.

Contour generation, feature extraction, the simulated datasets and the
known-truth coverage simulation were not repeated when the repository was
prepared. Their configurations, seeds and implementation are included. A future
complete image-to-results replay should report its computational cost and
validate the new stage manifests before claiming full regeneration.

## Scope changes in the published tree

The tree follows the submitted manuscript and supplement. Superseded manuscript
plans, change logs, unused figure builders, duplicate rendered figures and an
unreported fixed-severity branch were removed earlier, with the four tests
specific to that branch.

The two exploratory CT cohorts were removed from the study. Their rows were
removed from the supplied tables, their macros from the number record and their
panels from the supplementary figures; the MRI records are unchanged. The locked
files that name them are unchanged ([DATA.md](DATA.md#scope-of-the-locked-files)).
One interface test now expects the MRI counts, and the threshold replay compares
the MRI rows of a fresh run.

`provenance/BASELINE.json` identifies the original snapshot, `ORIGIN.json` its
source and export ledger, `REPRODUCIBILITY.json` binds the current files to the
original inputs and numerical expressions, and `CT_REMOVAL.json` records the
removal. `provenance/VALIDATION.json` records the validation of the tree before
the CT removal. The original protocol notes are in `provenance/protocol_notes/`.
The commit identifiers in these records refer to the authors' development
repository, which is not public; the public repository begins with the released
tree. `verify` checks the recorded file hashes against the current checkout.
