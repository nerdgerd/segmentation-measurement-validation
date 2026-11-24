"""Audit measurement-unit conversions on a fixed empirical descriptor battery.

The transforms depend on descriptor identities and source reference SD, never on
target performance. Whole-battery moment controls are accompanied by direct-value
checks on the fixed first eight descriptor identifiers when matrices are supplied.
"""

from dataclasses import replace
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
from math import isfinite, isnan, nan, sqrt
from typing import Mapping

import numpy as np
import pandas as pd

from .construction import MatrixMeasurements
from .empirical_math import aggregate_patient_moments
from .moments import (
    JointMoments,
    ReferenceMoments,
    _joint_from_components,
    compare_disagreement,
    hierarchical_weights,
)
from .policies import PRIMARY_POLICIES, construct_screens
from .provenance import canonical_bytes, content_hash
from .transport import fit_source_model, predict_target, weighted_quantile


DIMENSIONLESS_TOLERANCE = 1e-10
DIRECT_DESCRIPTOR_COUNT = 8
SCALE_EXPONENTS = (-6, -3, -1, 1, 3, 6)


def descriptor_transform(
    cfg: dict, descriptor_id: str, source_sd: float
) -> dict[str, object]:
    if (
        not isinstance(descriptor_id, str)
        or not descriptor_id
        or not isfinite(source_sd)
        or source_sd <= 0
    ):
        raise ValueError(
            "Unit control needs a stable descriptor identity and positive source reference SD"
        )
    key = [
        cfg["randomness"]["master_seed"],
        cfg["randomness"]["namespace"],
        "unit_conversion_control",
        descriptor_id,
    ]
    digest = hashlib.sha256(canonical_bytes(key)).digest()
    exponent = SCALE_EXPONENTS[digest[0] % len(SCALE_EXPONENTS)]
    sign = -1 if digest[1] % 2 else 1
    shift_coefficient = digest[2] % 9 - 4
    scale = sign * 10.0**exponent
    return {
        "scale": scale,
        "shift": scale * shift_coefficient * source_sd,
        "source_sd": source_sd,
        "shift_coefficient": shift_coefficient,
        "transform_key_sha256": digest.hex(),
    }


def affine_joint_moments(
    moments: JointMoments, scale: float, shift: float
) -> JointMoments:
    if not isfinite(scale) or scale == 0 or not isfinite(shift):
        raise ValueError(
            "A unit conversion requires finite nonzero scale and finite shift"
        )
    converted = _joint_from_components(
        scale * moments.reference_mean + shift,
        scale**2 * moments.reference_variance,
        scale * moments.disagreement_mean,
        scale**2 * moments.disagreement_variance,
        scale**2 * moments.reference_disagreement_covariance,
    )
    if moments.ccc_reason is not None:
        converted = replace(converted, ccc=nan, ccc_reason=moments.ccc_reason)
    return converted


def _equal(before: float, after: float) -> tuple[bool, float]:
    if isnan(before) or isnan(after):
        return isnan(before) and isnan(after), nan
    if not isfinite(before) or not isfinite(after):
        return before == after, 0.0 if before == after else float("inf")
    difference = abs(before - after)
    return difference <= DIMENSIONLESS_TOLERANCE, difference


def _decimal_equal(before: Decimal, after: Decimal) -> tuple[bool, Decimal]:
    if before.is_nan() or after.is_nan():
        return before.is_nan() and after.is_nan(), Decimal("NaN")
    if not before.is_finite() or not after.is_finite():
        return before == after, Decimal(0) if before == after else Decimal("Infinity")
    difference = abs(before - after)
    return difference <= Decimal(str(DIMENSIONLESS_TOLERANCE)), difference


def _precise_slope_control(
    synthetic: JointMoments, observed: JointMoments, scale: float, shift: float
) -> dict:
    """Check slopes independently before/after conversion at unchanged tolerance.

    Large dimensionless slopes can have binary64 spacing above the absolute
    gate. Promote the exact stored inputs before transforming or dividing, and
    compare before casting display values back to binary64. This controls only
    verification arithmetic; reported scientific diagnostics remain unchanged.
    """
    fields = (
        "reference_mean",
        "reference_variance",
        "disagreement_mean",
        "disagreement_variance",
        "reference_disagreement_covariance",
        "disagreement_mse",
        "ccc",
    )

    def promote(moments):
        return replace(
            moments,
            **{
                field: Decimal.from_float(float(getattr(moments, field)))
                for field in fields
            },
        )

    def slope(moments):
        variance = moments.reference_variance
        if not variance.is_finite() or variance <= 0:
            return Decimal("NaN")
        return moments.reference_disagreement_covariance / variance

    with localcontext() as context:
        context.prec = 80
        context.traps[InvalidOperation] = False
        synthetic_decimal, observed_decimal = promote(synthetic), promote(observed)
        before = slope(synthetic_decimal) - slope(observed_decimal)
        decimal_scale, decimal_shift = (
            Decimal.from_float(float(scale)),
            Decimal.from_float(float(shift)),
        )
        converted_synthetic = affine_joint_moments(
            synthetic_decimal, decimal_scale, decimal_shift
        )
        converted_observed = affine_joint_moments(
            observed_decimal, decimal_scale, decimal_shift
        )
        after = slope(converted_synthetic) - slope(converted_observed)
        passed, change = _decimal_equal(before, after)
        return {
            "value_before": float(before),
            "value_after": float(after),
            "expected_after": float(before),
            "absolute_change": float(change),
            "passed": passed,
            "value_before_precision_text": str(before),
            "value_after_precision_text": str(after),
            "absolute_change_precision_text": str(change),
            "verification_arithmetic": "decimal80_exact_binary64_inputs",
        }


def _scores(
    cfg, source_moments, target_reference, synthetic_moments, source_patient_counts
):
    scores = {policy: {} for policy in PRIMARY_POLICIES}
    for descriptor, moments in source_moments.items():
        model = fit_source_model(
            moments,
            independent_patients=source_patient_counts[descriptor],
            min_independent_patients=cfg["features"]["min_source_independent_patients"],
            residual_roundoff_relative_tolerance=cfg["remedy"][
                "residual_variance_negative_roundoff_relative_tolerance"
            ],
        )
        scores["source_ccc"][descriptor] = moments.ccc
        scores["synthetic_ccc"][descriptor] = synthetic_moments[descriptor].ccc
        scores["error_transport"][descriptor] = predict_target(
            model, target_reference[descriptor]
        ).ccc
    if any(
        not isfinite(value)
        for scores_by_descriptor in scores.values()
        for value in scores_by_descriptor.values()
    ):
        raise ValueError(
            "Unit audit must receive the locked universe with all three construction scores defined"
        )
    return scores


def _value_matrix(matrix: MatrixMeasurements, descriptors, transforms=None):
    if not set(descriptors).issubset(matrix.descriptor_ids):
        raise ValueError(
            "A direct-value matrix lacks a descriptor in the fixed control subset"
        )
    columns = [matrix.descriptor_ids.index(descriptor) for descriptor in descriptors]
    reference, alternate = (
        matrix.reference[:, columns].copy(),
        matrix.alternate[:, columns].copy(),
    )
    if transforms is not None:
        scale = np.array(
            [transforms[descriptor]["scale"] for descriptor in descriptors], dtype=float
        )
        shift = np.array(
            [transforms[descriptor]["shift"] for descriptor in descriptors], dtype=float
        )
        reference, alternate = reference * scale + shift, alternate * scale + shift
    return MatrixMeasurements(
        tuple(descriptors),
        matrix.keys.copy(),
        reference,
        alternate,
        matrix.eligible[:, columns].copy(),
    )


def audit_unit_conversions(
    cfg: dict,
    *,
    direction: str,
    scope: str,
    source_moments: Mapping[str, JointMoments],
    target_reference: Mapping[str, ReferenceMoments],
    synthetic_moments: Mapping[str, JointMoments],
    observed_moments: Mapping[str, JointMoments],
    diagnostic_synthetic_moments: Mapping[str, JointMoments],
    source_patient_counts: Mapping[str, int],
    matrices: Mapping[str, MatrixMeasurements] | None = None,
) -> pd.DataFrame:
    """Return auditable empirical controls; the caller supplies the evaluation gate.

    ``matrices`` may contain source, target_reference, synthetic, and observed inputs.
        ``synthetic_moments`` are frozen construction scores. The separate diagnostic
        synthetic/observed moments already use declared matched support; this audit
        does not replace construction scores with their evaluation-support versions.
    """
    descriptors = tuple(sorted(source_moments))
    if not descriptors or any(
        set(mapping) != set(descriptors)
        for mapping in (target_reference, synthetic_moments, source_patient_counts)
    ):
        raise ValueError(
            "Unit audit inputs must share the locked construction descriptor universe"
        )
    if set(observed_moments) - set(descriptors):
        raise ValueError(
            "Observed unit-control descriptors are outside the locked universe"
        )
    if set(diagnostic_synthetic_moments) - set(descriptors):
        raise ValueError(
            "Diagnostic synthetic descriptors are outside the locked universe"
        )
    transforms = {
        descriptor: descriptor_transform(
            cfg, descriptor, sqrt(source_moments[descriptor].reference_variance)
        )
        for descriptor in descriptors
    }
    converted_source, converted_synthetic, converted_reference = {}, {}, {}
    for descriptor in descriptors:
        scale, shift = transforms[descriptor]["scale"], transforms[descriptor]["shift"]
        converted_source[descriptor] = affine_joint_moments(
            source_moments[descriptor], scale, shift
        )
        converted_synthetic[descriptor] = affine_joint_moments(
            synthetic_moments[descriptor], scale, shift
        )
        original = target_reference[descriptor]
        converted_reference[descriptor] = ReferenceMoments(
            scale * original.mean + shift, scale**2 * original.variance
        )
    scores = _scores(
        cfg, source_moments, target_reference, synthetic_moments, source_patient_counts
    )
    converted_scores = _scores(
        cfg,
        converted_source,
        converted_reference,
        converted_synthetic,
        source_patient_counts,
    )
    before_selections, after_selections = (
        construct_screens(**scores),
        construct_screens(**converted_scores),
    )
    rows = []
    for descriptor in descriptors:
        for policy in PRIMARY_POLICIES:
            before, after = (
                scores[policy][descriptor],
                converted_scores[policy][descriptor],
            )
            passed, change = _equal(before, after)
            rows.append(
                {
                    "audit_component": "whole_battery_policy_score",
                    "input_role": policy,
                    "descriptor_id": descriptor,
                    **transforms[descriptor],
                    "value_before": before,
                    "value_after": after,
                    "absolute_change": change,
                    "passed": passed,
                    "status": "defined",
                }
            )
        observed = observed_moments.get(descriptor)
        diagnostic_synthetic = diagnostic_synthetic_moments.get(descriptor)
        if observed is None or diagnostic_synthetic is None:
            rows.append(
                {
                    "audit_component": "within_descriptor_disagreement",
                    "descriptor_id": descriptor,
                    **transforms[descriptor],
                    "passed": None,
                    "status": "matched_disagreement_moments_unavailable",
                }
            )
            continue
        scale, shift = transforms[descriptor]["scale"], transforms[descriptor]["shift"]
        comparison = compare_disagreement(diagnostic_synthetic, observed)
        converted_observed = affine_joint_moments(observed, scale, shift)
        converted_diagnostic = affine_joint_moments(diagnostic_synthetic, scale, shift)
        converted_comparison = compare_disagreement(
            converted_diagnostic, converted_observed
        )
        for metric in (
            "log2_rmse_ratio",
            "reference_slope_difference",
            "standardized_bias_difference",
        ):
            before, after = (
                getattr(comparison, metric),
                getattr(converted_comparison, metric),
            )
            expected = (
                np.sign(scale) * before
                if metric == "standardized_bias_difference"
                else before
            )
            passed, change = _equal(expected, after)
            if metric == "log2_rmse_ratio":
                passed = (
                    passed
                    and comparison.rmse_stratum == converted_comparison.rmse_stratum
                )
            precision = {}
            if metric == "reference_slope_difference":
                precision = _precise_slope_control(
                    diagnostic_synthetic, observed, scale, shift
                )
                precision.update(
                    {
                        "native_binary64_value_before": before,
                        "native_binary64_value_after": after,
                        "native_binary64_absolute_change": change,
                    }
                )
                passed = precision["passed"]
            rows.append(
                {
                    "audit_component": "within_descriptor_disagreement",
                    "input_role": metric,
                    "descriptor_id": descriptor,
                    **transforms[descriptor],
                    "value_before": before,
                    "value_after": after,
                    "expected_after": expected,
                    "absolute_change": change,
                    "rmse_stratum_before": comparison.rmse_stratum,
                    "rmse_stratum_after": converted_comparison.rmse_stratum,
                    "passed": passed,
                    "status": "undefined_preserved"
                    if isnan(before) and passed
                    else "defined",
                    **precision,
                }
            )
    for before, after in zip(before_selections, after_selections):
        rows.append(
            {
                "audit_component": "whole_battery_policy_selection",
                "input_role": before.policy,
                "selected_before": json_ids(before.selected_ids),
                "selected_after": json_ids(after.selected_ids),
                "passed": before.selected_ids == after.selected_ids,
                "status": "defined",
                "universe_count": len(descriptors),
                "budget": before.budget,
            }
        )
    if matrices is not None:
        allowed_roles = {"source", "target_reference", "synthetic", "observed"}
        if set(matrices) - allowed_roles:
            raise ValueError("Unknown direct empirical unit-control role")
        subset = descriptors[:DIRECT_DESCRIPTOR_COUNT]
        for role, matrix in matrices.items():
            original = _value_matrix(matrix, subset)
            transformed = _value_matrix(matrix, subset, transforms)
            minimum = (
                cfg["features"]["min_source_independent_patients"]
                if role == "source"
                else cfg["features"]["min_evaluation_independent_patients"]
                if role == "observed"
                else 1
            )
            coverage = (
                cfg["features"]["min_evaluation_weighted_support"]
                if role == "observed"
                else cfg["features"]["min_construction_weighted_availability"]
            )
            first = aggregate_patient_moments(
                original.patient_moments(),
                min_independent_patients=minimum,
                min_weighted_support=coverage,
            )
            second = aggregate_patient_moments(
                transformed.patient_moments(),
                min_independent_patients=minimum,
                min_weighted_support=coverage,
            )
            for descriptor, before, after in zip(subset, first.moments, second.moments):
                passed, change = _equal(before.ccc, after.ccc)
                passed = passed and before.ccc_reason == after.ccc_reason
                rows.append(
                    {
                        "audit_component": "direct_empirical_values_first_eight_ids",
                        "input_role": role,
                        "descriptor_id": descriptor,
                        **transforms[descriptor],
                        "value_before": before.ccc,
                        "value_after": after.ccc,
                        "absolute_change": change,
                        "passed": passed,
                        "status": before.ccc_reason or "defined",
                        "direct_subset_count": len(subset),
                    }
                )
        if "target_reference" in matrices:
            rows.extend(
                _raw_iqr_controls(matrices["target_reference"], descriptors, transforms)
            )
    else:
        rows.append(
            {
                "audit_component": "direct_empirical_values_first_eight_ids",
                "status": "not_run_no_matrices_supplied",
                "passed": None,
            }
        )
    result = pd.DataFrame(rows)
    result["direction"], result["scope"] = direction, scope
    result["dimensionless_tolerance"] = DIMENSIONLESS_TOLERANCE
    result["transform_manifest_sha256"] = content_hash(transforms)
    result["unit_control_specification"] = "audit/UNIT_CONTROL_SPECIFICATION.md"
    return result


def json_ids(values) -> str:
    return canonical_bytes(list(values)).decode("utf-8")


def _raw_iqr_controls(matrix, descriptors, transforms):
    values_before, values_after = {}, {}
    for descriptor in descriptors:
        column = matrix.descriptor_ids.index(descriptor)
        eligible = matrix.eligible[:, column]
        support = hierarchical_weights(
            matrix.keys.patient_id,
            matrix.keys.unit_id,
            matrix.keys.observation_id,
            eligible=eligible,
        )
        if not eligible.any():
            values_before[descriptor] = values_after[descriptor] = nan
            continue
        values = matrix.reference[eligible, column]
        weights = np.asarray(support.weights)[eligible]
        converted = (
            transforms[descriptor]["scale"] * values + transforms[descriptor]["shift"]
        )
        values_before[descriptor] = weighted_quantile(
            values, weights, 0.75
        ) - weighted_quantile(values, weights, 0.25)
        values_after[descriptor] = weighted_quantile(
            converted, weights, 0.75
        ) - weighted_quantile(converted, weights, 0.25)
    finite = [
        descriptor
        for descriptor in descriptors
        if isfinite(values_before[descriptor]) and isfinite(values_after[descriptor])
    ]
    before_rank = {
        descriptor: rank
        for rank, descriptor in enumerate(
            sorted(
                finite, key=lambda descriptor: (-values_before[descriptor], descriptor)
            ),
            start=1,
        )
    }
    after_rank = {
        descriptor: rank
        for rank, descriptor in enumerate(
            sorted(
                finite, key=lambda descriptor: (-values_after[descriptor], descriptor)
            ),
            start=1,
        )
    }
    return [
        {
            "audit_component": "raw_weighted_reference_iqr_unit_sensitivity",
            "input_role": "target_reference",
            "descriptor_id": descriptor,
            **transforms[descriptor],
            "value_before": values_before[descriptor],
            "value_after": values_after[descriptor],
            "rank_before": before_rank.get(descriptor),
            "rank_after": after_rank.get(descriptor),
            "rank_changed": before_rank.get(descriptor) != after_rank.get(descriptor),
            "passed": None,
            "status": "descriptive_unit_sensitivity_not_an_invariance_gate",
            "ranked_denominator": len(finite),
        }
        for descriptor in descriptors
    ]
