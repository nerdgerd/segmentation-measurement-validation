# Mathematical definitions

## Feature values from one contour

For one feature, X is the corrected-contour value and W the value from an
original or a generated contour. Define the feature difference δ = W − X. Under
the fixed patient and contour weights, write v = Var(X), a = Cov(X, δ),
b = E(δ), q = Var(δ) and M = q + b², the mean squared difference.

Lin's concordance correlation coefficient is

```text
CCC = 2(v + a) / [2(v + a) + M].
```

The denominator must be positive and all moments finite. Observed and generated
CCC values calculated from the same corrected contours share v, which can raise
both. Experiment 1 uses the Spearman correlation of CCC values across eligible
features, not across patients. See [moments.py](../study/src/redesign2/moments.py)
and [empirical_math.py](../study/src/redesign2/empirical_math.py).

## Simulated correlation baseline (Experiment 1)

The corrected-contour value X is normally distributed. The three simulation
models describe random differences, offset and proportional differences, and
random differences whose scale depends on X. Normal values can be negative; these
are models of the mechanism, not of anatomical feature distributions.

Family and feature fields are fixed before patients are simulated. The log error
scales of the generated differences are θ u_observed + sqrt(1 − θ²) u_independent,
with θ = 0 for independent and θ = 0.9 for strongly associated error scales. The
realised correlation across a finite feature set need not equal θ.

The median and IQR of log(v/μ²) are matched first, then each side's log(M/μ²).
Means are one in the dimensionless model. A common multiplier scales offset,
proportional effect and residual SD together. The first root on the fixed
601-point grid from 0 to 12 is used, with the recorded Brent refinement; a cell
without a root remains unmatched.

Exact model values describe an infinitely large cohort. Simulated datasets keep
the cohorts' patient counts, contours per patient and feature availability, with
100 generated contours per simulated patient and 2,000 datasets per setting. The
simulated correlation baseline is the median of the three exact model values at
θ = 0 within each target cohort and analysis. The held-out baseline uses
test-patient moments and is context for the training-fixed CCC values. See
[mechanism.py](../study/src/redesign2/mechanism.py),
[simulation.py](../study/src/redesign2/simulation.py) and
[finite_ranking.py](../study/experiments/finite_ranking.py).

## Magnitude of feature differences (Experiment 2)

```text
RMSE = sqrt(M)
L = log₂(RMSE_generated / RMSE_observed)
  = 0.5 [log₂(M_generated) − log₂(M_observed)].
```

L = −1 means half the observed RMSE, L = 0 equal RMSE and L = 1 twice the
observed RMSE. Median L and median |L| are reported separately. Exact zeros have
their own strata: both zero is undefined, a generated-only zero is −∞ and an
observed-only zero is +∞. Non-finite or invalid M is undefined. These cases are
counted, not folded into a finite median.

Multiplying both feature values by the same non-zero unit conversion leaves the
CCC and L unchanged; raw dispersions in different units are not comparable. The
calculation is [measurement_difference_ratio](../study/src/redesign2/moments.py),
and the complete comparison is `compare_disagreement` in the same module.

## Feature selection (Experiment 3)

A feature is retained when its CCC is at least the threshold t; a CCC equal to
the threshold counts as retained. The thresholds are 0.75, 0.80, 0.85, 0.90 and
0.95. The selection with observed CCC values is the reference.

| Count | Generated or source CCC | Observed CCC |
|---|---|---|
| Agreement | Retained | Retained |
| False retention | Retained | Not retained |
| False exclusion | Not retained | Retained |
| Agreement | Not retained | Not retained |

For a method's selection Q and the observed selection R, the false-retention
rate is |Q \ R| / |Q|, the false-exclusion rate |R \ Q| / |R|, the Jaccard
agreement |Q ∩ R| / |Q ∪ R| and the total disagreement |Q △ R| / |J|, where J is
the set of eligible features. Empty denominators remain undefined, and an
undefined CCC makes the whole cell undefined. Both methods use the same observed
selection and the same patient resampling indices. See
[threshold_decisions.py](../study/src/redesign2/threshold_decisions.py).

Bootstrap ranges are conditional stability summaries; the prespecified coverage
check failed. The top-quarter ranking endpoint is a separate supplementary
analysis and does not replace the threshold comparison.
