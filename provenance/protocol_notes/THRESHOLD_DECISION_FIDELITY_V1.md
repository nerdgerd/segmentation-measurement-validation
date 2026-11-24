# Amendment: threshold decision fidelity as Experiment 3

Date: 2026-09-21  
Status: choices locked before execution of the new endpoint  
Authority: `docs/14_CODEX_HANDOVER_THRESHOLD_DECISION_EXPERIMENT.md`

## Reason for the amendment

Review of the substantially revised manuscript identified a closer decision-level question than the existing fixed-budget ranking endpoint. Experiment 3 will now evaluate whether synthetic target contours and the available paired source contours produce the same descriptor retain/reject decisions as observed target contour pairs across five fixed CCC cutoffs. The prior top-quarter ranking analysis remains an unchanged supplementary sensitivity analysis.

This is a review-driven post hoc amendment to a completed retrospective study. Historical target results are known. It is neither an external preregistration nor an investigator-blinded analysis.

## Choices fixed before execution

- Cutoffs: 0.75, 0.80, 0.85, 0.90, and 0.95.
- A descriptor is retained when CCC is greater than or equal to the cutoff.
- The reference is the finite-sample observed target decision, not ground truth.
- Candidates are synthetic target CCC and observed source CCC. Contrasts are synthetic minus source.
- Full and audit analyses remain separate; reciprocal directions are never pooled.
- MRI is principal evidence. CT remains exploratory.
- The fixed eligible descriptor universe, CCC implementation, patient-first weighting, descriptor definitions, and eligibility rules are inherited from the locked study.
- Full-cohort resampling recomputes observed-target and synthetic-target CCCs from a shared target-patient bootstrap while holding source scores, severities, the generator, and the realized synthetic pool fixed.
- Audit resampling holds discovery-derived synthetic decisions and source decisions fixed and recomputes only observed-target CCCs on resampled audit patients.
- The bootstrap uses 2,000 patient-level replicates in the new `redesign2_threshold_decision_fidelity_v1` SHA-256 seed namespace.
- A point or bootstrap cell with any required undefined CCC is undefined; no descriptor is silently rejected, replaced, or removed after thresholding.
- Percentile endpoints are conditional bootstrap sampling-stability ranges. A range is withheld when more than 5% of replicates are undefined. No p-values are added.
- Primary outputs include retained counts, TP/FP/FN/TN, false-retention proportion, false-exclusion proportion, Jaccard agreement, overall disagreement rate, net retained-count difference, paired contrasts, and the complete descriptor crossing table.
- The existing top-quarter results must reproduce exactly before the new manuscript endpoint is accepted.

The machine-readable specification is `config/threshold_decision_fidelity_v1.yaml`.

## Immutable inputs

The experiment may read the completed `redesign2_v1` preparation, generation, construction, commitment, and evaluation artifacts. It may not regenerate images, contours, descriptor measurements, or synthetic contour pools. Every inherited file used by the run will be hashed in the new manifest.

## Interpretation limits

The experiment measures cutoff-decision fidelity for geometric descriptors under the observed alternate-contour processes. It does not establish anatomical truth, clinical utility, downstream predictive performance, or a universally appropriate CCC cutoff.
