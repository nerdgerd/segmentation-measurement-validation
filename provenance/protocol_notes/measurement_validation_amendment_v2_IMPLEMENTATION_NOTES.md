# Additional implementation notes

Before generic performance was calculated, inspection of the historical IQR
implementation identified NumPy's linear-interpolated raw unit percentiles.
The queued implementation's initial weighted inverse-CDF draft was replaced
with that exact inherited policy. IQR therefore retains both its mixed-unit
limitation and unequal nodule contribution; it is the specified historical
comparator, not a hierarchical reliability estimator. CCC, CV and utility
retain equal-patient/equal-unit weights. This explicit exception to the general
weighting description follows the requirement to preserve the inherited IQR
policy; no G4 evaluation outcome informed the choice.

Archived MRI binary descriptor-definedness is fully known from the current
explicit finite-value guard: finite is defined; nonfinite is undefined.
It is not inferred from the observed missingness rate. Historical MRI G4 calibrates
WT, TC, ET and present NETC separately, all to the fixed requested Dice. Its
intercompartment and volume fields come from the WT-calibrated contour. Saved
volumes therefore recover anatomical subtypes only for WT/intercompartment
entries; nonfinite other-region entries retain an explicitly unavailable subtype.
They remain undefined, with no imputation or convenient exclusion. The unknown
subtype does not leave binary pool membership unknown. Missing binary membership
or calibration identities would still block reuse.

Each descriptor retains an individual-contour measurement and its within-unit
draw variance. The historical MRI descriptor vector does not come from one
coherent partition, unlike primary S. This recovered semantic difference is
reported, rather than changing the inherited G4 calibration or claiming that the
two constructions differ only in available severity information. Calibration
identity includes region; every executed failed match remains included.

The matching root solver uses deterministic Brent refinement within the first
ascending frozen bracket, rather than the amendment's stated bisection. The
bracket, first-root rule, search grid and attained tolerance are unchanged; no
alternative root was selected. Both attained marginal checks are saved for every
cell. This is a numerical implementation deviation, not a changed matching law.

The current aggregation table's `support_changes` field is a conservative
completeness flag: it counts descriptors with any missing cell in the full pool,
not exact differences between eligible unit sets. The manuscript labels it pool
missingness; explicit complete-K patient counts and both CCC comparisons remain
in the immutable CSV. No computed value changes.

The brief's approximate source gain of 0.90--0.92 describes LIDC-to-NSCLC:
0.903 full and 0.922 audit. UPENN-to-TCGA instead has 0.700 full and 0.739
audit. A draft implementation note misassigned the 0.922 value; this wording
was corrected before generic evaluation began. The manuscript's generated
source-gain macros bind the correct cohort/scope cells, and all scopes remain.

An attempted addition to the first execution-audit Markdown changed a completed
stage's input bytes. The original prefix was restored only after its SHA-256
matched the recorded `reporting-correction-v1` input; additions live in this
separate file. No completed result or original audit content was changed.
