# Measurement validation amendment v2

Specified 2026-09-17, after the original locked analysis and the completed
measurement-level manuscript revision. Historical findings are already known.
All added analyses are descriptive. This is an execution lock, not external
preregistration or prospective confirmation. Authority: the project owner's
`docs/10_CODEX_HANDOVER_CRITICAL_REVISION.md`. The previous scientific records,
configurations, code and typesetting packages remain immutable.

## Inputs and scope

The original protocol/configuration is `config/protocol.lock.yaml`, SHA-256
`8139ac7fc83dcba4a345f3b76636bacede3497c3d503d19e32d05d9083504a8d`.
The accepted revised package is committed at `09842086`, with analytical source
commit `ce675fd8`. This amendment uses new configuration
`config/measurement_validation_amendment_v2.yaml` and a new output directory under
its canonical SHA-256. Before any new calculations, a lock JSON records the
amendment and configuration byte hashes. Every stage uses atomic RunWriter output,
exact input hashes, script copies, code state, environment and seed keys.

Original report inputs are under `runs/<original SHA>/study-report-unitprecision-v2/`:
`reference_evaluation_moments.csv`, `matched_observed_moments.csv`,
`synthetic_evaluation_moments.csv`, `descriptor_disagreement.csv`,
`descriptor_prediction_errors.csv`, `selected_set_utility.csv`,
`primary_and_contextual_contrasts.csv`, `profile_agreement.csv`,
`mechanism_coefficient_fields.csv`, `aggregation_audit.csv`,
`bootstrap_replicates.csv`, `bootstrap_summary.csv`, `bootstrap_failures.csv`,
`cohort_design.csv`, `draw_convergence.csv` and `draw_convergence_summary.csv`
where present. The prior coverage diagnosis and raw fixtures
`runs/<original SHA>/coverage-final/coverage_repetitions.csv` remain inputs.
Original `construct-<direction>-<full/discovery>-<100/200>/` source models,
synthetic moments and patient caches, and `evaluate-<direction>-<full/audit>-unitprecision-v2/`
observed/reference/synthetic patient caches supply recomputation. No legacy runner
is imported or executed.

Directions are UPENN→BraTS-TCGA, BraTS-TCGA→UPENN, NSCLC→LIDC and LIDC→NSCLC;
both CT directions remain exploratory. Full and the original independent patient
audit scopes remain distinct; audit construction uses discovery patients only.
Original eligible batteries and selections are primary. Each sensitivity carries
its own explicit battery identity. Reader/nodule weights remain equal patients,
equal eligible units within patient, equal eligible alternates within unit.
Neither low performance nor unstable estimates create an exclusion.

## 1. Two marginal matches and finite samples

The previous mechanism matched only reference spread. It will be called
reference-spread-matched, with its extreme construction supplementary. The new
mechanism matches the median and IQR of `log(v/mu^2)` and, separately for observed
and synthetic sides, `log(D/mu^2)`. `D` is the complete disagreement second moment,
including bias and reference dependence. Matching uses the original evaluation
moment tables and eligible descriptor IDs with finite positive `v,D` and finite
nonzero `mu`; excluded matching values receive explicit reasons. No sides are
pooled. Simulation zero points are fixed at source mean one. Reference log variance
is `median(log(v/mu^2)) + 2*h*(g-median(g))`, with
`h=IQR(log(v/mu^2))/(2*IQR(g))` using the inherited coefficient field.

For each law/side/theta, let `u` be the original coupled residual-scale field.
Starting from the complete original conditional Gaussian law, multiply **all**
disagreement coefficients (bias, proportional slope and residual SD) for descriptor
j by `exp((lambda-1)*u_j/2)`. The heteroscedastic log-SD slope stays fixed.
This changes `a`, bias and residual variance coherently and retains conditional
law structure; it is not residual-variance-only calibration. Compute exact
population D including all terms. Numerically match its log-IQR to the empirical
side's `log(D/mu^2)` IQR, searching lambda in [0,12] on 601 fixed grid points,
taking the first ascending sign-change and refining by deterministic bisection.
An exact grid match is accepted first; if no root exists, record unmatched and
do not choose another bound, field, pooled target or law. A single common factor
then matches the median log D, without changing its spread. Root tolerance is
1e-6. Save original/matched moments and both attained spreads/medians.

These marginal matches do not match joint reference/disagreement distributions,
descriptor-specific anatomy, covariance/bias/slope marginals or empirical latent
coupling. Theta remains a simulation coefficient-field input; it is never estimated
from real-to-real correlation. Failure or attenuation narrows the mechanism claim.

Natural-zero sensitivity is defined from descriptor definitions before outcomes:
include absolute volumes, radial/maximum diameter/thickness magnitude measures,
component counts, granulometric retained fractions/radii, nonnegative curvature
magnitudes, skeleton sizes/endpoints, spectral powers and fractional topology
measurements. Exclude all volume ratios, classical shape ratios, radial/thickness
CVs, curvature-class fractions derived from signed angular shape index, and all
intercompartment composites. Each ID and reason is written to a static audit table;
signed/angular/ratio semantics and near-zero-mean calibration flags are separate.
Recompute matching and population mechanism for this subset; no subset is chosen
according to the resulting association. Full/audit original batteries remain primary.

Finite simulation: all eight direction/scope cells, three laws and theta {0,0.9},
2,000 datasets per matched cell. Use actual patient/unit/reader metadata, original
eligible descriptor subsets, original family fields and explicit observed/reference
and synthetic availability arrays, including audit patient restriction. Preserve
the inherited 100-draw convention. Reuse the existing hierarchical Gaussian field
generator and patient sufficient-moment estimator, with new seed keys including
direction/scope/law/theta/absolute replicate. Unmatched cells are reported blocked.
Save every estimated association, evaluable count and undefined fraction; summarize
median, IQR and central 95% range. No rho>=0.5 rule is a headline endpoint.

## 2. Empirical permutation baseline

Use matched observed and synthetic moments on each complete eligible battery.
Condition on target reference moments. Permute the complete **dimensionless**
synthetic tuple `(a/v, D/v, b/sqrt(v), slope=1+a/v)` as one unit, then map all
components onto the receiving descriptor's reference variance. This avoids mixed
measurement units and preserves moment feasibility. The raw reference moments and
observed tuple stay fixed. Reconstruct synthetic CCC as
`2*(1+a/v)/(2*(1+a/v)+D/v)`. Marginal tuple components are never permuted separately.

MRI repeated-region descriptor stems are clusters: remove the region prefix from
regional descriptor IDs and carry their entire available region vector together.
Blocks are `(audited family, exact available region signature)`; nonregional
volume/intercompartment IDs are separate single-vector clusters with their audited
region signature. CT uses the same rule on its single region. A block of size one
is fixed and explicitly reported. Meaningful blocks require at least two clusters.
Save each member, cluster count, region signature and factorial permutation count;
total distinct mappings are the product of block factorials, reported exactly as
an integer string. Shared descriptor/region clustering is therefore preserved.

Use 10,000 Monte Carlo draws, identity allowed, per cell. An unrestricted descriptor
permutation is a separately labelled sensitivity with 10,000 draws. Report original
association, null median/IQR/95% range, tie-aware empirical percentile
`(count(null<observed)+0.5*count(null==observed))/defined`, and the symmetric full
source-real benchmark (audit source/target context reported separately if available).
This conditional correspondence baseline is not an exact test or estimator of
latent theta. No formal p-values or exchangeability claims are made.

## 3. Coverage prediction

Retain the failed locked bias gate unchanged. For all 16 stored utility endpoints
and 12 paired contrasts, use truth, mean estimate, empirical SD (ddof=1), issuance
and midpoint diagnostics. Calculate `r=abs(bias)/SD` and
`Phi(1.959964-2*r)-Phi(-1.959964-2*r)`. Save predicted minus observed coverage
(all-dataset coverage primary; issued-only coverage also retained), signed residuals,
law/template/endpoint type and undefined reasons. Explain the normal-location
approximation and approximate doubling of centre bias. Describe correlation,
mean absolute residual and min/max residual without significance tests. Important
residual discrepancies remain visible. No new interval estimator or changed gate.

## 4. Inherited G4 and information restriction

Recovered historical specification: `config/analysis_config_stage_g_v11.yaml`,
configuration SHA `92e360451be37907b545903fc268c6ae959641c776fb0d3f5b5f0535a12a08cc`,
framework configuration SHA
`7911b8d936ab854e9c87e3d141bac88ad4bc9af5d52370e49ad3549c87c21b95`,
master seed 20260905, namespace `stage_g_transport_v1`, generating commit
`25d26f61beb7d87f445318a762eca08c9654f57c`, Diffuse sigma=6 voxels,
fixed whole-tumour Dice=0.85, five ensembles of 20 draws. Reference-only crop,
original bisection bounds/tolerance/iterations and seed keys are inherited exactly.
Verify all four generation manifests and parquet SHA identities, configuration,
input manifests, contour identities, descriptor conventions and reference values.
Historical source metadata does not authorize reading source pairs in construction.

Reuse archived descriptor rows only if compatible with the current selected
reference and geometry. Reconstruct current explicit descriptor-definedness from
saved calibration statuses and anatomy, without inferring a convenient rule from
finite values. If compatibility cannot be proved, that cohort's G4 evaluation is
blocked pending exact regeneration; never substitute source-donor draws or severity.
Record any missing recovery setting as a blocker before construction.

Construction/ranking is a separate function/process that receives only target
reference measurements, generic synthetic rows, static battery metadata and the
historical configuration. It cannot receive target alternates, real CCC or paired
source data. Evaluate sealed selections afterwards against random and target oracle,
target-reference IQR (inherited comparator), and CV as a labelled sensitivity.
IQR is descending raw reference IQR, preserved as the inherited policy with an
explicit mixed-unit limitation; it is not presented as a unit-invariant ranking.
CV=SD/abs(mean), undefined at zero/nonfinite mean, is descending sensitivity only.
Budget and eligibility are current full/audit rules; audit ranking uses discovery.
Save U, G, selected IDs, precision counts/chance, regret and all undefined/precision
flags. Use 2,000 conditional patient bootstraps with full re-ranking and audit fixed
selections, preserving the original 5% undefined issuance rule. No calibrated
superiority claim. Missing historical pool-definedness is a blocker, not imputation.

## 5. Static denominator audit and failure reasons

Before sensitivity outcomes, examine the admitted extraction definitions. List all
denominators and why eligibility guarantees positivity or why zero/instability is
possible. Frozen risk list: all five MRI `volratio_*` descriptors; regional
`spharm_L*`/`spharm_rough` (degree-zero coefficient can vanish for centroid-ray
geometry), regional `obb_fill` (degenerate extent product), regional `solidity` and
`surface_convexity` (degenerate mesh/hull volume/area), and intercompartment
`rim_thickness_cv`, `edema_thickness_cv`, `edema_anisotropy` (potentially zero
measured thickness/mean under admitted support). Exclude these definition classes
in both MRI/CT wherever present, before examining sensitivity utility. Every
included/excluded denominator classification is saved with source function/guard.
Guards guaranteed positive by nonempty voxel support (volume, equivalent radius,
radial mean, distance-transform mean, filled volume), fixed canonical radius,
clipped eigenvalues and guarded surface counts remain included. A numerical
epsilon is not anatomical positivity. Location features are outside current battery.

Repeat CCC association/calibration, matched disagreement and re-ranked D/S/T utility
on the remaining explicit battery, budget floor(J/4). No new extraction, change of
primary descriptors or revised primary result. Trace original and sensitivity
bootstrap replicates using the original exact empirical-bootstrap keys for replay,
with new keys only for genuinely new procedures. Preserve full re-ranking versus
audit fixed selections. Record every replicate/policy/descriptor failure, using
exclusive endpoint-level reasons plus all descriptor-level causes. Reason vocabulary:
undefined_descriptor_value, invalid_CCC_denominator, insufficient_patient_support,
undefined_source_ranking_score, undefined_synthetic_ranking_score,
incomplete_top_k, undefined_selected_target_utility,
zero_random_oracle_denominator, other_numerical_failure. Save counts by cell/policy,
affected IDs, risk-list membership and whether removal changes issuance. Match
replayed original U and S-D estimates to stored replicate values before using the
trace. Do not claim risk descriptors explain withheld ranges without this check.

## 6. Secondary diagnostics

Case mix: save `log2(v_target/v_source)` from target reference moments and inherited
source model variance, all directions/full/audit and each family/region. Summarize
median/IQR/95% range and optionally absolute shift versus absolute transport error
descriptively. No independent-descriptor p-values.

Recovery grid: keep inherited exact-law h=1 demonstration. Recalculate inherited
target variance multipliers {0.5,1,2,4} and mean shifts {0,1,2} for the observed-side
two-marginal-matched settings and both theta values, all three laws, plus the
original changed-law cell (variance multiplier 2, shift 1). Do not fit a rescue model.
Save algebraic exact recovery and violations under matched dispersion separately.

Finite pool: use existing earlier/final moment scores and replicate ensembles from
each final full/discovery construction. Compare available 50→100 or 100→200 pools,
ranking only within the original eligibility battery. Save final draws, triggering
precision diagnostics, both selected sets, overlap count/fraction, target U in the
relevant full/audit evaluation, both S-D effects and sign-change flag. These compare
nested available pools, not independent Monte Carlo replications. If earlier score
rows cannot be recovered, report that limitation instead of assuming stability.

Aggregation: recover current empirical descriptor-level single-contour and
draw-averaged CCC from the original aggregation audit, verify exact cohort,
descriptor and support conventions against current MRI primary estimates; save
differences and median/p95 by cohort and K. Historical alternate definitions are
not reused as current estimates. Full distributions supplementary.

Split half: 200 deterministic random patient permutations per cohort/scope,
first floor(N/2) versus remaining patients, retaining all their units/readers.
Recompute observed CCC with original min 10 eligible patients and 90% support
rules. Report all undefined cells (e.g. small audit CT halves), common-finite
descriptor association, evaluable counts and top-k overlap when complete halves
permit it. Save all split patient IDs. Split-half reproducibility contextualizes
target-ranking noise; it is not a theoretical ceiling for full-cohort methods.

## Figures, prose and decision-neutral reporting

Four main figures: (1) dominant paired-source measurement question and compact
generic information branch, (2) two marginal matches, empirical blocked null and
finite association distributions replacing threshold bars, (3) retained empirical
ordering/calibration/disagreement without tautological percentile-window counts,
(4) anchored selection with neutral S-D colour and consistent full/audit shapes.
Combine supplementary coverage prediction/case mix, sensitivity/failure reasons,
pool/split-half stability, matched-h transport and untrimmed mechanism/disagreement
diagnostics. Every display comes from saved source tables and retains all cells.

Rebalance abstract MRI effects; omit the one-direction G comparison. Report source
gain range without presenting G as independent evidence. Replace internal version
language with transparent post-analysis status. Rename undefined-replicate columns,
repair negative ranges and duplicate exploratory labels. Bridge primary literature
on probabilistic sampled maps, spatial uncertainty and downstream measurements;
verify primary sources/DOIs and maintain screening records. Separate AI research
assistance from writing assistance; authors must confirm patient-data disclosure,
human review/responsibility and missing names/ethics/funding/CRediT/release facts.
Do not invent these declarations. Draw the vector graphical abstract only after
claims stabilize, with no general-purpose generative image model.

Run relevant tests, independent calculation/seed/block/eligibility/leakage checks,
typeset main and supplement, inspect every rendered page and seal a new package.
Required records: `REVISION_CHANGELOG.md`, `RESULTS_PROVENANCE.md` and final
validation report. Narrow claims if matching weakens them. Coverage/generic/
definition sensitivities are retained regardless of direction. Exact unavailable
settings or incompatible archived data are reported blocked, never substituted.
