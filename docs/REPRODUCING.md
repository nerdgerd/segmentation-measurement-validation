# Reproducing the study

## 1. Install and verify the supplied results

Use Python 3.12 and run from the repository root:

```powershell
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
python tools/reproduce.py verify
python tools/reproduce.py summary
python tools/reproduce.py figures
python -m pytest -q
```

`verify` checks the recorded file identities, the unchanged numerical expressions
and all 40 feature-selection cells. `summary` prints the four principal rows.
`figures` rebuilds the 12 figures and records input, source and output hashes in
`study/outputs/paper_assets/FIGURE_MANIFEST.json`. This reference replay uses the
saved feature-level results; it is not a fresh patient-level computation.

Experiment 2 has an additional replay that recalculates every saved RMSE ratio
from the saved mean squared differences and writes a new stamped run:

```powershell
python tools/reproduce.py run measurement-differences --run-id q2-check
```

Completed run names cannot be reused.

## 2. Import the patient-level inputs

The full computations start from patient-level inputs that are not distributed:
audited patient and contour identities, prepared feature values, generated
contour pools, construction caches, evaluation tables and the original manifests.
The authors import them from the original study checkout:

```powershell
python tools/import_local_inputs.py --source <path-to-study-checkout>
```

The importer records hashes and explicit path mappings under `private_inputs/`,
which Git ignores. Large immutable files use hard links where supported; `--copy`
requests copies. Imaging locations are recorded, not copied. Missing assets are
listed in `private_inputs/IMPORT_RECORD.json`. Rebuilding these inputs from the
public imaging collections requires the recorded contour identities and the
locked protocol (see [DATA.md](DATA.md#scope-of-the-locked-files)).

## 3. Run a named analysis

```powershell
python tools/reproduce.py list
python tools/reproduce.py run ranking-baseline --run-id q1-models
python tools/reproduce.py run ranking-finite --run-id q1-finite --workers 1
python tools/reproduce.py run feature-selection --run-id q3-bootstrap
python tools/reproduce.py run contour-averaging --run-id averaging
python tools/reproduce.py run reciprocal-audit --run-id raw-audit
```

All supplementary command names are in [EXPERIMENTS.md](EXPERIMENTS.md).
`ranking-finite` accepts `--pilot` for a timing check with the same models, seed
keys and dataset count. A run name changes storage, not randomness.

The feature-selection run uses 2,000 patient bootstraps. To compare its outputs
with the supplied tables:

```powershell
python tools/verify_threshold_replay.py <path-to-new-threshold-run>
```

The comparison covers the MRI rows of all six tables and 152 columns, allowing
only the storage `run_id` to differ.

## 4. Recompute the shared contour stages

The stage command takes explicit prepared, generated and construction runs:

```powershell
python tools/stages.py prepare --run-id prepared --workers 1
python tools/stages.py generate --run-id draws --prepared-run <prepared> --source brats_tcga --target upenn --workers 1
python tools/stages.py construct --run-id prediction --prepared-run <prepared> --generation-run <draws> --source brats_tcga --target upenn --scope full
python tools/stages.py evaluate --run-id evaluation --prepared-run <prepared> --generation-run <draws> --construction-run <prediction> --source brats_tcga --target upenn --scope full --commitments <index.json> --coverage-run <coverage>
```

The bracketed arguments are the run directories printed by the earlier commands.
Construction uses the `full` and `discovery` scopes; evaluation uses `full` and
`audit`. The commitment index must list every commitment named in the locked
protocol before evaluation opens the original target contours. Coverage is run
with `tools/stages.py coverage` on the prepared run.

`generate --refine-from <initial> --convergence-run <construction>` applies only
the fixed one-time doubling rule. Changing cohort identities, availability,
templates or protocol parameters is a new analysis and needs a new specification
and seed namespace.

## Runtime

Reference replay and tests take seconds. Contour generation, the simulated
datasets and the coverage simulation take hours; run a pilot and consult the
recorded generation-cost table first. Installing or checking the repository never
starts those computations. [VALIDATION.md](VALIDATION.md) separates what has been
replayed from what was not repeated.
