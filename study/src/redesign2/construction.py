"""Construct directional screens from physically separated permitted products.

Source observed pairs may be opened; target alternate products never enter this
module's file allowlist. Discovery construction receives separate reference and
draw files, so audit patients cannot affect its universe, scores, or selections.
"""

from dataclasses import asdict, dataclass
from io import BytesIO
import json
from math import isfinite, nan
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .access import Asset, ConstructionInputs
from .empirical_math import (
    MomentProfile,
    PatientMomentTable,
    aggregate_patient_moments,
    prepare_patient_moments,
)
from .moments import ReferenceMoments, hierarchical_weights
from .policies import (
    PRIMARY_POLICIES,
    PolicySelection,
    construct_screens,
    selection_overlap,
)
from .provenance import RunWriter, content_hash, file_hash, inside
from .transport import fit_source_model, predict_target, target_extrapolation_fraction


@dataclass(frozen=True)
class MatrixMeasurements:
    descriptor_ids: tuple[str, ...]
    keys: pd.DataFrame
    reference: np.ndarray
    alternate: np.ndarray
    eligible: np.ndarray

    def patient_moments(self) -> PatientMomentTable:
        return prepare_patient_moments(
            self.reference,
            self.alternate,
            self.keys.patient_id.tolist(),
            self.keys.unit_id.tolist(),
            self.keys.observation_id.tolist(),
            eligible=self.eligible,
            descriptor_ids=self.descriptor_ids,
        )


@dataclass(frozen=True)
class ConstructionProducts:
    tables: dict[str, pd.DataFrame]
    selections: tuple[PolicySelection, ...]
    universe: tuple[str, ...]
    ready: bool
    reason: str | None
    draw_count: int


def _validate_long(
    frame: pd.DataFrame, *, cohort: str, role: str, descriptor_ids: Sequence[str]
) -> pd.DataFrame:
    required = {
        "cohort",
        "patient_id",
        "unit_id",
        "annotation_id",
        "role",
        "descriptor_id",
        "value",
        "definedness",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            "Prepared measurements do not implement the normalized long-table contract"
        )
    result = frame.copy()
    if result.empty or set(result.cohort) != {cohort} or set(result.role) != {role}:
        raise PermissionError(
            "Prepared measurement cohort or contour role does not match its capability"
        )
    for column in (
        "patient_id",
        "unit_id",
        "annotation_id",
        "descriptor_id",
        "definedness",
    ):
        if result[column].isna().any() or any(
            not isinstance(value, str) or not value for value in result[column]
        ):
            raise ValueError(
                f"Missing stable identity or declared definedness: {column}"
            )
    if set(result.descriptor_id) != set(descriptor_ids):
        raise ValueError(
            "Prepared descriptor identities differ from the audited compatible universe"
        )
    keys = ["patient_id", "unit_id", "annotation_id", "descriptor_id"]
    if result.duplicated(keys).any():
        raise ValueError("Duplicate contour/descriptor measurement")
    n_observations = len(result[keys[:-1]].drop_duplicates())
    if len(result) != n_observations * len(descriptor_ids):
        raise ValueError(
            "Incomplete contour/descriptor grid; missing cells require explicit definedness"
        )
    if (
        role == "reference"
        and (
            result.groupby(["patient_id", "unit_id"]).annotation_id.nunique() != 1
        ).any()
    ):
        raise ValueError(
            "A reference unit must contain exactly one selected annotation"
        )
    result["value"] = pd.to_numeric(result.value, errors="raise")
    if (~np.isfinite(result.value) & result.definedness.eq("defined")).any():
        raise ValueError("A measurement marked defined is nonfinite")
    return result


def source_pair_matrix(
    reference: pd.DataFrame,
    alternate: pd.DataFrame,
    *,
    cohort: str,
    descriptor_ids: Sequence[str],
) -> MatrixMeasurements:
    reference = _validate_long(
        reference, cohort=cohort, role="reference", descriptor_ids=descriptor_ids
    )
    alternate = _validate_long(
        alternate, cohort=cohort, role="alternate", descriptor_ids=descriptor_ids
    )
    join = ["patient_id", "unit_id", "descriptor_id"]
    if set(map(tuple, reference[join[:-1]].drop_duplicates().to_numpy())) != set(
        map(tuple, alternate[join[:-1]].drop_duplicates().to_numpy())
    ):
        raise ValueError("Source reference/alternate unit correspondence is incomplete")
    paired = alternate.merge(
        reference[join + ["annotation_id", "value", "definedness"]],
        on=join,
        suffixes=("_alternate", "_reference"),
        validate="many_to_one",
        how="left",
    )
    if (paired.annotation_id_alternate == paired.annotation_id_reference).any():
        raise ValueError(
            "An observed alternate cannot be the selected reference annotation"
        )
    paired["observation_id"] = paired.annotation_id_alternate
    keys = ["patient_id", "unit_id", "observation_id"]
    x = paired.pivot(
        index=keys, columns="descriptor_id", values="value_reference"
    ).reindex(columns=descriptor_ids)
    w = paired.pivot(
        index=keys, columns="descriptor_id", values="value_alternate"
    ).reindex(index=x.index, columns=descriptor_ids)
    paired["eligible"] = paired.definedness_reference.eq(
        "defined"
    ) & paired.definedness_alternate.eq("defined")
    eligible = paired.pivot(
        index=keys, columns="descriptor_id", values="eligible"
    ).reindex(index=x.index, columns=descriptor_ids)
    return MatrixMeasurements(
        tuple(descriptor_ids),
        x.index.to_frame(index=False),
        x.to_numpy(float),
        w.to_numpy(float),
        eligible.to_numpy(bool),
    )


def reference_matrix(
    reference: pd.DataFrame, *, cohort: str, descriptor_ids: Sequence[str]
) -> MatrixMeasurements:
    reference = _validate_long(
        reference, cohort=cohort, role="reference", descriptor_ids=descriptor_ids
    )
    reference["observation_id"] = reference.annotation_id
    keys = ["patient_id", "unit_id", "observation_id"]
    x = reference.pivot(index=keys, columns="descriptor_id", values="value").reindex(
        columns=descriptor_ids
    )
    reference["eligible"] = reference.definedness.eq("defined")
    eligible = reference.pivot(
        index=keys, columns="descriptor_id", values="eligible"
    ).reindex(index=x.index, columns=descriptor_ids)
    values = x.to_numpy(float)
    return MatrixMeasurements(
        tuple(descriptor_ids),
        x.index.to_frame(index=False),
        values,
        values.copy(),
        eligible.to_numpy(bool),
    )


def synthetic_matrix(
    reference: MatrixMeasurements,
    draws: pd.DataFrame,
    definedness: pd.DataFrame,
    *,
    cohort: str,
    draw_limit: int | None = None,
) -> MatrixMeasurements:
    """Join one reference to each distinct draw, never to real alternate readers."""
    keys = ["cohort", "patient_id", "unit_id", "draw_id", "batch"]
    required = set(keys) | set(reference.descriptor_ids)
    if not required.issubset(draws.columns) or not required.issubset(
        definedness.columns
    ):
        raise ValueError(
            "Synthetic values and definedness must implement the same wide descriptor schema"
        )
    if (
        draws.empty
        or set(draws.cohort) != {cohort}
        or set(definedness.cohort) != {cohort}
    ):
        raise PermissionError("Synthetic products belong to a different target cohort")
    data, flags = draws.copy(), definedness.copy()
    for column in ("patient_id", "unit_id"):
        if any(not isinstance(value, str) or not value for value in data[column]):
            raise ValueError(
                "Synthetic patient/unit identifiers must be stable nonempty strings"
            )
    for frame in (data, flags):
        numeric_ids = pd.to_numeric(frame.draw_id, errors="raise")
        if (
            not np.isfinite(numeric_ids).all()
            or not np.equal(numeric_ids, np.floor(numeric_ids)).all()
        ):
            raise ValueError("Synthetic draw identifiers must be integers")
        frame["draw_id"] = numeric_ids.astype(int)
        frame["batch"] = pd.to_numeric(frame.batch, errors="raise").astype(int)
        if frame.duplicated(["patient_id", "unit_id", "draw_id"]).any():
            raise ValueError(
                "Repeated synthetic draw identity; reader duplication is forbidden"
            )
    data = data.sort_values(keys, kind="stable").reset_index(drop=True)
    flags = flags.sort_values(keys, kind="stable").reset_index(drop=True)
    if not data[keys].equals(flags[keys]):
        raise ValueError("Synthetic values and explicit definedness are not aligned")
    reference_units = set(
        map(tuple, reference.keys[["patient_id", "unit_id"]].to_numpy())
    )
    draw_units = set(
        map(tuple, data[["patient_id", "unit_id"]].drop_duplicates().to_numpy())
    )
    if reference_units != draw_units:
        raise ValueError("Synthetic units differ from the permitted reference scope")
    draw_sets = [
        tuple(group.draw_id)
        for _, group in data.groupby(["patient_id", "unit_id"], sort=True)
    ]
    if any(ids != draw_sets[0] for ids in draw_sets):
        raise ValueError(
            "Every reference must have the same explicit draw pool; incomplete cases cannot be dropped"
        )
    if draw_limit is not None:
        if draw_limit < 1 or draw_limit > len(draw_sets[0]):
            raise ValueError("Unavailable predeclared draw subset")
        keep = data.draw_id.isin(draw_sets[0][:draw_limit])
        data, flags = (
            data.loc[keep].reset_index(drop=True),
            flags.loc[keep].reset_index(drop=True),
        )
    location = {
        (row.patient_id, row.unit_id): index for index, row in reference.keys.iterrows()
    }
    reference_indices = np.array(
        [location[(row.patient_id, row.unit_id)] for row in data.itertuples()],
        dtype=int,
    )
    columns = list(reference.descriptor_ids)
    values = data[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if flags[columns].isna().any().any() or any(
        not isinstance(value, str) or not value
        for value in flags[columns].to_numpy().ravel()
    ):
        raise ValueError("Every synthetic value needs an explicit definedness category")
    draw_eligible = flags[columns].eq("defined").to_numpy(bool)
    if np.any(draw_eligible & ~np.isfinite(values)):
        raise ValueError("A synthetic value marked defined is nonfinite")
    output_keys = data[["patient_id", "unit_id"]].copy()
    output_keys["observation_id"] = "draw:" + data.draw_id.astype(str)
    return MatrixMeasurements(
        reference.descriptor_ids,
        output_keys,
        reference.reference[reference_indices],
        values,
        reference.eligible[reference_indices] & draw_eligible,
    )


def _support_rows(profile: MomentProfile, prefix: str) -> list[dict[str, object]]:
    return [
        {
            "descriptor_id": descriptor,
            **{prefix + key: value for key, value in asdict(support).items()},
        }
        for descriptor, support in zip(profile.descriptor_ids, profile.support)
    ]


def _patient_cache_rows(table: PatientMomentTable) -> pd.DataFrame:
    rows = []
    fields = (
        "reference_mean",
        "reference_variance",
        "disagreement_mean",
        "disagreement_variance",
        "reference_disagreement_covariance",
        "eligible",
        "original_support_coverage",
        "eligible_units",
        "eligible_observations",
    )
    for i, patient in enumerate(table.patient_ids):
        for j, descriptor in enumerate(table.descriptor_ids):
            row = {"patient_id": patient, "descriptor_id": descriptor}
            row.update({field: getattr(table, field)[i, j].item() for field in fields})
            rows.append(row)
    return pd.DataFrame(rows)


def patient_cache_from_frame(
    frame: pd.DataFrame, *, descriptor_ids: Sequence[str]
) -> PatientMomentTable:
    """Decode a declared descriptor subset without inferring support or identities."""
    ids = tuple(descriptor_ids)
    fields = (
        "reference_mean",
        "reference_variance",
        "disagreement_mean",
        "disagreement_variance",
        "reference_disagreement_covariance",
        "eligible",
        "original_support_coverage",
        "eligible_units",
        "eligible_observations",
    )
    if (
        not ids
        or len(set(ids)) != len(ids)
        or not {"patient_id", "descriptor_id", *fields}.issubset(frame.columns)
    ):
        raise ValueError(
            "Incomplete patient-moment cache schema or descriptor universe"
        )
    if not set(ids).issubset(frame.descriptor_id):
        raise ValueError("A locked descriptor is absent from the patient-moment cache")
    selected = frame.loc[frame.descriptor_id.isin(ids)].copy()
    if selected.duplicated(["patient_id", "descriptor_id"]).any():
        raise ValueError("Duplicate patient/descriptor moment identity")
    if any(not isinstance(value, str) or not value for value in selected.patient_id):
        raise ValueError("Patient cache needs stable nonempty string identifiers")
    patients = tuple(sorted(set(selected.patient_id)))
    if len(selected) != len(patients) * len(ids):
        raise ValueError("Patient/descriptor cache grid is incomplete")

    def boolean(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError("Patient cache eligibility must be explicitly Boolean")

    selected["eligible"] = selected.eligible.map(boolean)
    arrays = []
    for field in fields:
        wide = selected.pivot(
            index="patient_id", columns="descriptor_id", values=field
        ).reindex(index=patients, columns=ids)
        values = (
            wide.to_numpy(bool)
            if field == "eligible"
            else wide.apply(pd.to_numeric, errors="raise").to_numpy(float)
        )
        if field != "eligible" and not np.isfinite(values).all():
            raise ValueError(
                "Patient cache contains nonfinite stored sufficient moments"
            )
        if field in {
            "reference_variance",
            "disagreement_variance",
            "original_support_coverage",
            "eligible_units",
            "eligible_observations",
        } and np.any(values < 0):
            raise ValueError(
                "Patient cache contains an invalid negative support/variance field"
            )
        if field == "original_support_coverage" and np.any(values > 1 + 1e-12):
            raise ValueError(
                "Patient cache support exceeds its original observation mass"
            )
        if field in {"eligible_units", "eligible_observations"}:
            if not np.equal(values, np.floor(values)).all():
                raise ValueError("Patient cache counts must be integers")
            values = values.astype(int)
        values.setflags(write=False)
        arrays.append(values)
    present = arrays[5]
    if not np.array_equal(present, arrays[7] > 0) or not np.array_equal(
        present, arrays[8] > 0
    ):
        raise ValueError(
            "Patient cache eligibility contradicts its declared unit/observation counts"
        )
    return PatientMomentTable(ids, patients, *arrays)


def construct_from_frames(
    cfg: dict,
    *,
    source: str,
    target: str,
    scope: str,
    descriptor_ids: Sequence[str],
    source_reference: pd.DataFrame,
    source_alternate: pd.DataFrame,
    target_reference: pd.DataFrame,
    draw_features: pd.DataFrame,
    draw_definedness: pd.DataFrame,
) -> ConstructionProducts:
    """Numerical construction with no argument for target alternate measurements."""
    if scope not in {"full", "discovery"}:
        raise ValueError("Construction scope must be full or discovery")
    if {"source": source, "target": target} not in cfg["directions"]:
        raise ValueError("Direction is outside the fixed protocol")
    ids = tuple(descriptor_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("An audited unique compatible descriptor universe is required")
    min_source = cfg["features"]["min_source_independent_patients"]
    coverage = cfg["features"]["min_construction_weighted_availability"]
    src = source_pair_matrix(
        source_reference, source_alternate, cohort=source, descriptor_ids=ids
    )
    tgt = reference_matrix(target_reference, cohort=target, descriptor_ids=ids)
    source_profile = aggregate_patient_moments(
        src.patient_moments(),
        min_independent_patients=min_source,
        min_weighted_support=coverage,
    )
    target_cache = tgt.patient_moments()
    target_profile = aggregate_patient_moments(
        target_cache, min_independent_patients=1, min_weighted_support=coverage
    )
    synthetic_full = synthetic_matrix(
        tgt, draw_features, draw_definedness, cohort=target
    )
    draw_count = int(draw_features.groupby(["patient_id", "unit_id"]).size().iloc[0])
    allowed_draw_counts = (
        cfg["synthetic"]["draw_count_initial"],
        cfg["synthetic"]["convergence"]["refined_draw_count"],
    )
    if draw_count not in allowed_draw_counts:
        raise ValueError(
            "Construction requires the initial or single permitted refined draw count"
        )
    synthetic_cache = synthetic_full.patient_moments()
    synthetic_profile = aggregate_patient_moments(
        synthetic_cache, min_independent_patients=1, min_weighted_support=coverage
    )
    comparison_count = (
        cfg["synthetic"]["convergence"]["compare_draw_counts"][0]
        if draw_count == allowed_draw_counts[0]
        else allowed_draw_counts[0]
    )
    earlier = synthetic_matrix(
        tgt, draw_features, draw_definedness, cohort=target, draw_limit=comparison_count
    )
    earlier_profile = aggregate_patient_moments(
        earlier.patient_moments(),
        min_independent_patients=1,
        min_weighted_support=coverage,
    )
    model_rows, prediction_rows, extrapolation_rows = [], [], []
    scores = {policy: {} for policy in PRIMARY_POLICIES}
    eligibility, reasons, precision_candidates = {}, {}, {}
    for index, descriptor in enumerate(ids):
        observed = source_profile.moments[index]
        model = fit_source_model(
            observed,
            independent_patients=source_profile.support[index].independent_patients,
            min_independent_patients=min_source,
            residual_roundoff_relative_tolerance=cfg["remedy"][
                "residual_variance_negative_roundoff_relative_tolerance"
            ],
        )
        target_moments = target_profile.moments[index]
        prediction = predict_target(
            model,
            ReferenceMoments(
                target_moments.reference_mean, target_moments.reference_variance
            ),
        )
        model_rows.append(
            {
                "descriptor_id": descriptor,
                **asdict(model),
                "source_ccc": observed.ccc,
                **{
                    "source_" + key: value
                    for key, value in asdict(observed).items()
                    if key not in ("ccc", "reference_mean", "reference_variance")
                },
            }
        )
        prediction_rows.append({"descriptor_id": descriptor, **asdict(prediction)})
        scores["source_ccc"][descriptor] = observed.ccc
        scores["synthetic_ccc"][descriptor] = synthetic_profile.moments[index].ccc
        scores["error_transport"][descriptor] = prediction.ccc
        descriptor_reasons = [
            value
            for value in (
                source_profile.support[index].reason,
                model.reason,
                target_profile.support[index].reason,
                prediction.reason,
            )
            if value
        ]
        precision_candidates[descriptor] = not descriptor_reasons
        if synthetic_profile.support[index].reason:
            descriptor_reasons.append(synthetic_profile.support[index].reason)
        if not all(isfinite(scores[policy][descriptor]) for policy in PRIMARY_POLICIES):
            descriptor_reasons.append("undefined_primary_construction_score")
        reasons[descriptor] = ";".join(dict.fromkeys(descriptor_reasons)) or None
        eligibility[descriptor] = not descriptor_reasons
        source_mask, target_mask = src.eligible[:, index], tgt.eligible[:, index]
        if source_mask.any() and target_mask.any():
            source_weights = hierarchical_weights(
                src.keys.patient_id,
                src.keys.unit_id,
                src.keys.observation_id,
                eligible=source_mask,
            )
            target_weights = hierarchical_weights(
                tgt.keys.patient_id,
                tgt.keys.unit_id,
                tgt.keys.observation_id,
                eligible=target_mask,
            )
            diagnostic = target_extrapolation_fraction(
                src.reference[source_mask, index],
                np.asarray(source_weights.weights)[source_mask],
                tgt.reference[target_mask, index],
                np.asarray(target_weights.weights)[target_mask],
                quantiles=tuple(cfg["remedy"]["source_support_quantiles"]),
            )
            extrapolation_rows.append(
                {"descriptor_id": descriptor, **diagnostic, "reason": None}
            )
        else:
            extrapolation_rows.append(
                {
                    "descriptor_id": descriptor,
                    "source_lower_quantile": nan,
                    "source_upper_quantile": nan,
                    "target_weighted_fraction_outside": nan,
                    "reason": "no_eligible_source_or_target_references",
                }
            )
    universe = tuple(
        sorted(descriptor for descriptor in ids if eligibility[descriptor])
    )
    provisional = (
        construct_screens(
            **{
                policy: {
                    descriptor: scores[policy][descriptor] for descriptor in universe
                }
                for policy in PRIMARY_POLICIES
            }
        )
        if universe
        else ()
    )
    differences, convergence_rows = [], []
    for index, descriptor in enumerate(ids):
        first, last = (
            earlier_profile.moments[index].ccc,
            synthetic_profile.moments[index].ccc,
        )
        finite = isfinite(first) and isfinite(last)
        difference = abs(first - last) if finite else nan
        if precision_candidates[descriptor] and finite:
            differences.append(difference)
        convergence_rows.append(
            {
                "descriptor_id": descriptor,
                "earlier_draw_count": comparison_count,
                "final_draw_count": draw_count,
                "earlier_synthetic_ccc": first,
                "final_synthetic_ccc": last,
                "absolute_ccc_change": difference,
                "in_construction_universe": eligibility[descriptor],
                "precision_candidate": precision_candidates[descriptor],
                "comparison_defined": finite,
            }
        )
    median = float(np.median(differences)) if differences else nan
    p95 = float(np.quantile(differences, 0.95)) if differences else nan
    limits = cfg["synthetic"]["convergence"]
    inadequate = bool(differences) and (
        median > limits["median_absolute_ccc_change_limit"]
        or p95 > limits["p95_absolute_ccc_change_limit"]
    )
    needs_refinement = inadequate and draw_count == allowed_draw_counts[0]
    reason = (
        "empty_construction_universe"
        if not universe
        else "undefined_draw_precision_comparison"
        if not differences
        else "draw_refinement_required"
        if needs_refinement
        else None
    )
    ready = reason is None
    overlap = nan
    earlier_map = earlier_profile.ccc_mapping()
    if universe and all(isfinite(earlier_map[descriptor]) for descriptor in universe):
        earlier_selections = construct_screens(
            source_ccc={
                descriptor: scores["source_ccc"][descriptor] for descriptor in universe
            },
            synthetic_ccc={
                descriptor: earlier_map[descriptor] for descriptor in universe
            },
            error_transport={
                descriptor: scores["error_transport"][descriptor]
                for descriptor in universe
            },
        )
        overlap = selection_overlap(earlier_selections[1], provisional[1])
    convergence_summary = {
        "earlier_draw_count": comparison_count,
        "final_draw_count": draw_count,
        "universe_count": len(universe),
        "precision_candidate_count": sum(precision_candidates.values()),
        "defined_comparisons": len(differences),
        "undefined_comparisons_in_candidate_battery": sum(precision_candidates.values())
        - len(differences),
        "median_absolute_ccc_change": median,
        "p95_absolute_ccc_change": p95,
        "synthetic_selection_overlap": overlap,
        "refinement_required": needs_refinement,
        "precision_unresolved_after_allowed_refinement": inadequate
        and draw_count == allowed_draw_counts[1],
    }
    ranking = {
        (selection.policy, descriptor): rank
        for selection in provisional
        for rank, descriptor in enumerate(selection.ranked_ids, start=1)
    }
    selected = {
        (selection.policy, descriptor)
        for selection in provisional
        for descriptor in selection.selected_ids
    }
    score_rows = [
        {
            "policy": policy,
            "descriptor_id": descriptor,
            "score": scores[policy][descriptor],
            "eligible": eligibility[descriptor],
            "eligibility_reason": reasons[descriptor],
            "rank": ranking.get((policy, descriptor)),
            "selected": ready and (policy, descriptor) in selected,
        }
        for policy in PRIMARY_POLICIES
        for descriptor in ids
    ]
    selection_rows = [
        {
            "policy": selection.policy,
            "descriptor_id": descriptor,
            "rank": rank,
            "score": selection.ranked_scores[rank - 1],
            "budget": selection.budget,
            "universe_count": len(universe),
            "universe_sha256": content_hash(list(universe)),
        }
        for selection in provisional
        if ready
        for rank, descriptor in enumerate(selection.selected_ids, start=1)
    ]
    tables = {
        "source_models": pd.DataFrame(model_rows),
        "source_support": pd.DataFrame(_support_rows(source_profile, "source_")),
        "target_predictions": pd.DataFrame(prediction_rows),
        "source_extrapolation": pd.DataFrame(extrapolation_rows),
        "target_reference_support": pd.DataFrame(
            _support_rows(target_profile, "target_reference_")
        ),
        "synthetic_support": pd.DataFrame(
            _support_rows(synthetic_profile, "synthetic_")
        ),
        "synthetic_moments": pd.DataFrame(
            [
                {"descriptor_id": descriptor, **asdict(moments)}
                for descriptor, moments in zip(ids, synthetic_profile.moments)
            ]
        ),
        "construction_scores": pd.DataFrame(score_rows),
        "selections": pd.DataFrame(
            selection_rows,
            columns=[
                "policy",
                "descriptor_id",
                "rank",
                "score",
                "budget",
                "universe_count",
                "universe_sha256",
            ],
        ),
        "draw_convergence": pd.DataFrame(convergence_rows),
        "draw_convergence_summary": pd.DataFrame([convergence_summary]),
        "target_reference_patient_moments": _patient_cache_rows(target_cache),
        "synthetic_patient_moments": _patient_cache_rows(synthetic_cache),
    }
    direction = f"{source}_to_{target}"
    for table in tables.values():
        table["direction"], table["scope"] = direction, scope
    return ConstructionProducts(
        tables, provisional if ready else (), universe, ready, reason, draw_count
    )


def _manifest(path: Path, cfg: dict) -> dict:
    data = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if data["protocol_sha256"] != content_hash(cfg):
        raise ValueError("Input run belongs to a different protocol configuration")
    if data["status"] != "complete":
        raise ValueError("Input run is not complete")
    return data


def construct_direction(
    workspace,
    cfg,
    prepared_run,
    generation_run,
    *,
    source: str,
    target: str,
    scope: str,
    run_id: str,
    descriptor_manifest=None,
) -> Path:
    """Read only allowlisted products and write an immutable construction result."""
    workspace = Path(workspace).resolve()
    prepared_run, generation_run = (
        Path(prepared_run).resolve(),
        Path(generation_run).resolve(),
    )
    direction = f"{source}_to_{target}"
    if {"source": source, "target": target} not in cfg["directions"] or scope not in {
        "full",
        "discovery",
    }:
        raise ValueError("Unspecified direction or construction scope")
    prepared_meta, generated_meta = (
        _manifest(prepared_run, cfg),
        _manifest(generation_run, cfg),
    )
    extra = generated_meta.get("extra", {})
    if extra.get("source") != source or extra.get("target") != target:
        raise PermissionError(
            "Generated source severity/target direction does not match construction"
        )
    manifest_name = descriptor_manifest or cfg["workspace"].get("descriptor_manifest")
    if not manifest_name:
        raise ValueError("Audited descriptor manifest remains unresolved")
    descriptor_path = Path(manifest_name)
    descriptor_path = (
        descriptor_path
        if descriptor_path.is_absolute()
        else workspace / descriptor_path
    )
    descriptor_path = inside(workspace, descriptor_path)
    descriptor_metadata = pd.read_csv(descriptor_path, dtype=str, keep_default_na=False)
    required_metadata = {
        "feature_id",
        "domain",
        "family",
        "region",
        "units",
        "definition",
    }
    if not required_metadata.issubset(descriptor_metadata.columns):
        raise ValueError("Audited compatible descriptor metadata is incomplete")
    modality = cfg["cohorts"][source]["modality"]
    if cfg["cohorts"][target]["modality"] != modality:
        raise ValueError("Cross-modality construction is outside the protocol")
    metadata = descriptor_metadata.loc[descriptor_metadata.domain.eq(modality)]
    if metadata.empty or metadata.feature_id.duplicated().any():
        raise ValueError("Audited compatible descriptor metadata is incomplete")
    if (metadata[list(required_metadata)] == "").any().any():
        raise ValueError("An audited descriptor scientific field is unresolved")
    prefix = "discovery_" if scope == "discovery" else ""
    specs = {
        "source_reference": (
            prepared_run,
            prepared_meta,
            f"{source}/reference_features.csv",
            "source_product",
            "source",
        ),
        "source_alternate": (
            prepared_run,
            prepared_meta,
            f"{source}/alternate_features.csv",
            "source_product",
            "source",
        ),
        "target_reference": (
            prepared_run,
            prepared_meta,
            f"{target}/{prefix}reference_features.csv",
            "target_reference",
            scope,
        ),
        "draw_features": (
            generation_run,
            generated_meta,
            f"{prefix}draw_features.csv",
            "synthetic",
            scope,
        ),
        "draw_definedness": (
            generation_run,
            generated_meta,
            f"{prefix}draw_definedness.csv",
            "synthetic",
            scope,
        ),
    }
    assets = {}
    for name, (root, manifest, relative, role, asset_scope) in specs.items():
        if relative not in manifest["outputs"]:
            raise ValueError(
                f"Required separated input product is absent from its manifest: {relative}"
            )
        assets[name] = Asset(
            inside(root, root / relative),
            manifest["outputs"][relative],
            direction,
            role,
            asset_scope,
        )
    inputs = ConstructionInputs(direction, scope, assets)
    frames = {
        name: pd.read_csv(
            BytesIO(inputs.read_bytes(name)),
            float_precision="round_trip",
            dtype={
                "cohort": str,
                "patient_id": str,
                "unit_id": str,
                "annotation_id": str,
                "descriptor_id": str,
                "definedness": str,
            },
        )
        for name in specs
    }
    products = construct_from_frames(
        cfg,
        source=source,
        target=target,
        scope=scope,
        descriptor_ids=metadata.feature_id.tolist(),
        **frames,
    )
    input_paths = [asset.path for asset in assets.values()] + [
        descriptor_path,
        prepared_run / "run_manifest.json",
        generation_run / "run_manifest.json",
    ]
    writer = RunWriter(
        workspace,
        cfg,
        "construction",
        run_id,
        inputs=input_paths,
        estimand="fixed_budget_reference_adapted_screening_scores_before_target_alternate_evaluation",
        role=f"construction:{direction}:{scope}",
        status="complete" if products.ready else "blocked",
    )
    for name, table in products.tables.items():
        writer.write_table(name + ".csv", table)
    writer.write_json(
        "construction_status.json",
        {
            "direction": direction,
            "scope": scope,
            "ready_for_commitment": products.ready,
            "reason": products.reason,
            "universe": list(products.universe),
            "universe_sha256": content_hash(list(products.universe)),
            "draw_count": products.draw_count,
            "source_fits": "fixed_full_source",
            "target_alternates_accessed": False,
            "access_record": inputs.access_record,
        },
    )
    refinement_required = products.reason == "draw_refinement_required"
    status = (
        "refinement_required"
        if refinement_required
        else "construction_complete"
        if products.ready
        else "construction_unavailable"
    )
    writer.write_json(
        "status.json",
        {
            "status": status,
            "refinement_required": refinement_required,
            "source": source,
            "target": target,
            "scope": scope,
            "draw_count": products.draw_count,
            "reason": products.reason,
        },
    )
    return writer.finish(
        {
            "source": source,
            "target": target,
            "direction": direction,
            "scope": scope,
            "ready_for_commitment": products.ready,
            "reason": products.reason,
            "universe_count": len(products.universe),
            "draw_count": products.draw_count,
            "refinement_required": refinement_required,
            "status": status,
        }
    )
