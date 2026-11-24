# Validating generated segmentation variability for radiomic feature selection

Code, locked specifications and feature-level results for the article
*Validating generated segmentation variability for radiomic feature selection:
correlation, scale and decision agreement* (G. Heilemann, F. Eckert, D. Georg,
L. Zimmermann).

Contour perturbations are often validated by correlating feature-wise CCC values
from generated contours with those from observed contour pairs. This study shows
that such a correlation can arise from shared between-patient variance alone,
and that it can coexist with a mismatched magnitude of feature differences and
poor feature selection. It uses paired original and expert-corrected
glioblastoma contours from 147 UPENN-GBM and 97 BraTS-TCGA-GBM MRI patients and
192 geometric features. Perturbations calibrated on one cohort are applied to the
corrected contours of the other, and vice versa.

| Experiment | Question | Start with the code |
|---|---|---|
| 1: Correlation of CCC values | Does the correlation of generated with observed CCC values exceed a matched simulated correlation baseline? | [feature_ranking.py](study/experiments/feature_ranking.py), [mechanism.py](study/src/redesign2/mechanism.py) |
| 2: Magnitude of feature differences | How does the generated RMSE compare with the observed RMSE, feature by feature? | [measurement_differences.py](study/experiments/measurement_differences.py), [moments.py](study/src/redesign2/moments.py) |
| 3: Feature selection | Do generated CCC values select the same features as observed CCC values at thresholds 0.75–0.95? | [feature_selection.py](study/experiments/feature_selection.py), [threshold_decisions.py](study/src/redesign2/threshold_decisions.py) |

## Reproduce the supplied results

From this checkout, using Python 3.12:

```powershell
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
python tools/reproduce.py verify
python tools/reproduce.py summary
python tools/reproduce.py figures
python -m pytest -q
```

`verify` checks the recorded file identities and numerical code, then
reconstructs all 40 feature-selection cells (two directions, two analyses, five
thresholds, generated and source CCC values). `summary` prints the four principal
rows. `figures` rebuilds the 4 main and 8 supplementary figures under
`study/outputs/paper_assets/`; in the recorded environment they are
byte-identical to the submitted figures.
These commands use the supplied results and need no patient data.

## Understand or recompute an experiment

1. [Study overview](docs/STUDY.md): cohorts, contour roles, a worked example and the three experiments.
2. [All experiments](docs/EXPERIMENTS.md): each input, entry point, numerical helper and output.
3. [Reproduction instructions](docs/REPRODUCING.md): reference replay and full computational runs.
4. [Data and calculation rules](docs/DATA.md): weighting, support, missingness and the locked specifications.
5. [Mathematical definitions](docs/MATHEMATICS.md): CCC, simulations, RMSE ratios and feature selection.
6. [Validation](docs/VALIDATION.md): what the current code has reproduced.

Full computations start from patient-level inputs that are not distributed; see
the reproduction instructions.

## Repository layout

```text
study/experiments/       callable experiments in manuscript order
study/src/redesign2/     measurement, generation, statistics and simulation code
study/figures/           one pipeline for the main and supplementary figures
study/reference/         feature-level results and plotting inputs
study/config/            locked computational specifications and seed namespaces
study/tests/             scientific contracts and reproduction-interface checks
tools/                   command lines and the local input importer
docs/                    study, experiment and reproduction guides
provenance/              input identities, protocol notes and verification records
```

The locked specifications are distributed unchanged because their hashes
identify the analysis. They were written for a protocol that also included two
exploratory CT cohorts, which are not part of this study; no CT results, figures
or inputs are distributed ([DATA.md](docs/DATA.md#scope-of-the-locked-files)).

## Data

No imaging or patient-level data are included. The MRI data and segmentations
are available from The Cancer Imaging Archive under their own terms:
UPENN-GBM ([Bakas et al., 2022](https://doi.org/10.1038/s41597-022-01560-7)) and
the BraTS-TCGA-GBM segmentation labels
([Bakas et al., 2017](https://doi.org/10.1038/sdata.2017.117);
[dataset](https://doi.org/10.7937/K9/TCIA.2017.KLXWJJ1Q)).

## Licence and citation

The code and the supplied result tables are released under the [MIT licence](LICENSE).
The imaging collections keep their own licences and data-use terms. Citation
metadata are in [CITATION.cff](CITATION.cff).
