# Empirical unit-conversion verification

This implementation addendum was written before redesigned target performance
evaluation. It fixes a numerical control required by experiments M/E; it does not
amend the scientific hypotheses, feature universe, or locked protocol YAML.

## Fixed transformations

For each descriptor, hash the canonical JSON array
`[master_seed, namespace, "unit_conversion_control", descriptor_id]` with SHA-256.
Use digest byte 0 modulo 6 to select an exponent from `[-6, -3, -1, 1, 3, 6]`.
Digest byte 1 selects positive scale when even and negative scale when odd.
The absolute scale is `10**exponent`. Digest byte 2 modulo 9 minus 4 selects the
integer shift coefficient. The offset is
`scale * shift_coefficient * source_reference_SD`.

Apply the same affine conversion to source references/alternates and target
references/synthetic/observed alternates for that descriptor. Descriptor identity
and source reference SD determine the conversion. Target performance does not.

## Required numerical checks

1. Transform moments throughout the complete locked directional battery, refit the
   same source linear projection, and reconstruct its target CCC. Check all three
   construction score profiles and exact selected descriptor identities.
2. Check within-descriptor synthetic/observed RMSE ratios and their zero strata,
   reference slope differences, and signed standardized bias differences. A
   negative conversion reverses the signed bias diagnostic; nonsigned metrics
   retain their values. Undefined quantities must retain their undefined status.
3. Where actual matrices are supplied, convert values directly for the first eight
   lexicographically sorted stable descriptor identifiers, or all identifiers if
   fewer than eight. Recompute the same weighted moments with unchanged explicit
   eligibility and compare CCC and its definedness. The subset is fixed by IDs.
4. Report original and converted weighted reference-IQR ranks across the fixed
   battery when target reference values are available. Use inverse weighted
   empirical-CDF quartiles and stable-identifier ties. This describes sensitivity
   to arbitrary units; a rank change is not a required finding.

The numerical absolute tolerance for dimensionless CCC, log2 RMSE ratio, slope
difference, and signed bias comparisons is `1e-10`. This is an implementation
verification tolerance, not a scientific equivalence or practical-utility margin.
Policy selections and zero/undefined strata must agree exactly.

## Access and reporting

The caller must pass the all-direction construction commitment gate before supplying
observed target measurements. This helper performs no reads or writes. Report every
attempted control, failures, and unavailable direct-value checks in
`unit_conversion_control`, with the ordinary run manifest, source-table identities,
direction/scope, and transformation key/hash. A successful numerical control does
not identify anatomical error or establish freedom from case-mix dependence.
