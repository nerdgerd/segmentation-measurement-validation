# Study overview

The study asks whether generated contours can replace observed contour pairs when
deciding which geometric features are reliable. It evaluates three properties
separately: the correlation of CCC values across features, the magnitude of
feature differences and the features selected at a CCC threshold.

## Contours and patients

The data are paired original and expert-corrected glioblastoma contours from 147
UPENN-GBM and 97 BraTS-TCGA-GBM patients. The corrected contour is the reference;
the original contour preceded its correction. Observed feature differences are
original-contour minus corrected-contour values. Generated contours are
perturbations of the corrected contours. The two cohorts' original contours came
from different segmentation workflows; neither contour pair is an independent
expert repeat, and the corrected contour is not anatomical truth.

There are 192 geometric MRI features: volume, shape and extended geometric
features, each calculated for one region or a pair of regions (enhancing tumour,
non-enhancing tumour core, surrounding non-enhancing FLAIR hyperintensity, and the
tumour-core and whole-tumour unions). Intensity, texture, molecular and clinical variables
are not included.

## Worked example: TCGA to UPENN

1. Measure the Dice overlap between each original and corrected TCGA contour.
2. Sample a target Dice value from those TCGA values (the source cohort).
3. Perturb each corrected UPENN contour with the fixed Diffuse generator until
   its Dice overlap matches the sampled value within 0.005.
4. Calculate every eligible feature from every generated contour separately.
5. Calculate the generated CCC values against the corrected UPENN values.
6. At evaluation only, calculate the observed CCC values and feature differences
   from the withheld original UPENN contours.
7. Compare the generated CCC values, and the source CCC values already available
   from the TCGA pairs, with the observed CCC values.

UPENN-to-TCGA exchanges the roles. It reuses the same two cohorts and is not an
independent replication. Each direction started with 100 generated contours per
corrected contour; the prespecified precision check doubled UPENN-to-TCGA to 200.

## All-patient and held-out analyses

The all-patient analysis uses every target patient: the corrected contours to
generate contours and the original contours to evaluate them. The held-out
analysis fixes feature eligibility and generated CCC values in training patients
and evaluates them in separate test patients (TCGA 48/49, UPENN 73/74). Feature
eligibility is fixed within each direction and analysis; held-out UPENN-to-TCGA
has 191 eligible features. The two analyses are never pooled.

## The three experiments

**Experiment 1 — correlation of CCC values.** The Spearman correlation between
generated and observed CCC values across features is compared with a simulated
correlation baseline. Observed and generated CCC values calculated from the same
corrected contours share the between-patient variance, so they correlate even
when their error scales are independent. Three simulation models (random,
offset and proportional, and measurement-dependent random differences) are
matched to each target cohort; exact model values for an infinitely large cohort
and 2,000 simulated datasets per setting are reported. The baseline is the median
of the three exact values with independent error scales. A within-block
permutation analysis answers a separate, descriptive question.

**Experiment 2 — magnitude of feature differences.** For each feature,
L = log₂(generated RMSE / observed RMSE). Median L shows whether generated
differences are generally smaller or larger; median |L| prevents over- and
underestimation from cancelling.

**Experiment 3 — feature selection.** At CCC thresholds 0.75, 0.80, 0.85, 0.90
and 0.95, features selected with generated CCC values, and with source CCC
values, are compared with those selected with observed CCC values: false
retentions, false exclusions, Jaccard agreement and total disagreement. A
top-quarter ranking endpoint is a secondary analysis in the supplement.

## Scope of interpretation

Generator calibration never used the original target contours. The later
simulation matching used target-cohort summaries after the generated contours
had been evaluated, but not the feature-by-feature correspondence. This is a
retrospective reanalysis in which earlier results were known; it was not
externally preregistered. The held-out simulation baseline describes test-patient
moments; it does not simulate the whole training-to-test procedure.

The prespecified interval coverage check failed. Reportable patient-bootstrap
ranges describe conditional stability, with source quantities and generated
contour pools fixed, and are not calibrated confidence intervals. Coverage of the
feature-selection metrics was not separately validated. The four
direction/analysis results support no fitted trend.
