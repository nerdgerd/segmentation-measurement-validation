# M1 analytical counterexample: identical profile association at different coupling

This is an analytical completion of the counterexample requested by `docs/03_EXPERIMENTS.md`, M1. It does not add a simulation cell to the locked grid, change a seed, or use empirical target values. A numerical realization would require its own labeled source table and provenance; this note alone is a proof, not a reported computed result.

## Construction

Use the fixed demonstration coefficient fields already specified for the additive regime. Let `g_j` be the reference log-dispersion coefficients and `q_p,j > 0` the disagreement variances, for `p` in `{real, synthetic(theta=0), synthetic(theta=0.9)}`. Use the same real and independent synthetic coefficient fields at both coupling values. The two synthetic processes can have different descriptor disagreement parameters.

Assume the fixed `g_j` are distinct, and order descriptors so `g_1 < ... < g_J`. Define

`H = max(0, max over p and adjacent j of log(q_p,j+1 / q_p,j) / (2*(g_j+1 - g_j)))`.

For any `h > H`, and in particular the deterministic choice `h = H + 1`, the reference variances are `v_j = exp(2*h*g_j)`. Their relative spread is sufficient that, for every listed process and every adjacent pair,

`q_p,j+1 / v_j+1 < q_p,j / v_j`.

Since `CCC_p,j = 1 / (1 + q_p,j/(2*v_j))`, all three CCC profiles have the same strictly increasing descriptor order. Thus the real-versus-synthetic Spearman association is exactly one at both `theta=0` and `theta=0.9`.

The proof uses a finite maximum over a finite positive-variance battery. It does not require trying multiple seeds. If the fixed coefficient battery has equal `g` values, this particular construction is undefined and must be reported as such; do not silently perturb ties or substitute another battery.

## Interpretation and numerical precautions

This establishes that the same **profile association** can coexist with different imposed residual log-scale coupling. It does not establish identical per-descriptor CCC values, equal screening utility, equal disagreement parameters, or an empirical relationship between profile association and a latent anatomical quantity.

The threshold is derived to prove an existence claim. It is not an empirical estimate, a new default dispersion setting, a tuned primary result, or a claim that the locked finite grid must exhibit perfect association. Retain all original grid outputs beside the proof.

For a numerical source table, store `g`, each `log(q)`, the adjacent inequalities, `H`, and the chosen `h`. Compute log disagreement-to-reference ratios directly. Very large `h` can make floating-point CCC values round to one and create artificial numerical ties; the exact ordering should then be documented through the strict log-ratio inequalities. Do not force rounded CCC ranks to equal the analytical result.

The theorem is allowed by the already-locked request to construct an M1 counterexample, provided it is labeled as this analytical construction and never merged into the predeclared population or finite-sample grid. It does not authorize further data-generating regimes, seed search, or favorable empirical selection.
