"""Cache patient-conditional moments so resampling never expands contour draws.

Adapters declare every eligible cell and every original observation identity.
Conditional weights are rebuilt at each patient/unit/observation level. Original
support coverage is retained separately from the conditional joint moments.
"""

from dataclasses import dataclass, replace
from math import nan
from typing import Sequence

import numpy as np

from .moments import (
    JointMoments,
    _joint_from_components,
    hierarchical_weights,
    weighted_joint_moments,
)


@dataclass(frozen=True)
class DescriptorSupport:
    independent_patients: int
    eligible_units: int
    eligible_observations: int
    original_weighted_support: float
    reason: str | None = None

    @property
    def evaluable(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class MomentProfile:
    descriptor_ids: tuple[str, ...]
    moments: tuple[JointMoments, ...]
    support: tuple[DescriptorSupport, ...]

    def ccc_mapping(self) -> dict[str, float]:
        return {
            identifier: moments.ccc
            for identifier, moments in zip(self.descriptor_ids, self.moments)
        }


@dataclass(frozen=True)
class PatientMomentTable:
    descriptor_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    reference_mean: np.ndarray
    reference_variance: np.ndarray
    disagreement_mean: np.ndarray
    disagreement_variance: np.ndarray
    reference_disagreement_covariance: np.ndarray
    eligible: np.ndarray
    original_support_coverage: np.ndarray
    eligible_units: np.ndarray
    eligible_observations: np.ndarray


def prepare_patient_moments(
    reference: np.ndarray,
    alternate: np.ndarray,
    patient_ids: Sequence[str],
    unit_ids: Sequence[str],
    observation_ids: Sequence[str],
    *,
    eligible: np.ndarray,
    descriptor_ids: Sequence[str],
) -> PatientMomentTable:
    """Prepare reusable conditional moments without computing a ranking."""
    x, w, mask = (
        np.asarray(reference, dtype=float),
        np.asarray(alternate, dtype=float),
        np.asarray(eligible),
    )
    descriptors = tuple(descriptor_ids)
    if x.ndim != 2 or x.shape != w.shape or x.shape != mask.shape:
        raise ValueError(
            "Reference, alternate, and eligibility must be aligned row-by-descriptor matrices"
        )
    if mask.dtype != np.dtype(bool):
        raise TypeError("Eligibility must be an explicit Boolean matrix")
    if (
        len(descriptors) != x.shape[1]
        or len(set(descriptors)) != len(descriptors)
        or not descriptors
    ):
        raise ValueError("Unique descriptor identities must align with matrix columns")
    if any(not isinstance(value, str) or not value for value in descriptors):
        raise ValueError("Descriptor identities must be nonempty strings")
    if not (len(patient_ids) == len(unit_ids) == len(observation_ids) == x.shape[0]):
        raise ValueError("Observation metadata must align with matrix rows")
    hierarchical_weights(
        patient_ids, unit_ids, observation_ids, eligible=[True] * len(patient_ids)
    )
    if np.any(mask & (~np.isfinite(x) | ~np.isfinite(w))):
        raise ValueError(
            "An eligible measurement is nonfinite; record its failure explicitly"
        )
    patients = tuple(sorted(set(patient_ids)))
    if not patients:
        raise ValueError("The declared original patient support is empty")
    shape = (len(patients), len(descriptors))
    components = [np.zeros(shape, dtype=float) for _ in range(5)]
    present = np.zeros(shape, dtype=bool)
    coverage = np.zeros(shape, dtype=float)
    units = np.zeros(shape, dtype=int)
    observations = np.zeros(shape, dtype=int)
    row_groups = {patient: [] for patient in patients}
    for row, patient in enumerate(patient_ids):
        row_groups[patient].append(row)
    for patient_index, patient in enumerate(patients):
        rows = np.asarray(row_groups[patient], dtype=int)
        patient_units = [unit_ids[row] for row in rows]
        patient_observations = [observation_ids[row] for row in rows]
        for descriptor_index in range(len(descriptors)):
            kept = mask[rows, descriptor_index]
            support = hierarchical_weights(
                [patient] * len(rows),
                patient_units,
                patient_observations,
                eligible=kept,
            )
            coverage[patient_index, descriptor_index] = (
                support.original_support_coverage
            )
            units[patient_index, descriptor_index] = support.eligible_units
            observations[patient_index, descriptor_index] = (
                support.eligible_observations
            )
            if not support.eligible_observations:
                continue
            selected_rows = rows[kept]
            weights = np.asarray(support.weights)[kept]
            moments = weighted_joint_moments(
                x[selected_rows, descriptor_index],
                w[selected_rows, descriptor_index],
                weights,
            )
            present[patient_index, descriptor_index] = True
            for array, value in zip(
                components,
                (
                    moments.reference_mean,
                    moments.reference_variance,
                    moments.disagreement_mean,
                    moments.disagreement_variance,
                    moments.reference_disagreement_covariance,
                ),
            ):
                array[patient_index, descriptor_index] = value
    for array in (*components, present, coverage, units, observations):
        array.setflags(write=False)
    return PatientMomentTable(
        descriptors, patients, *components, present, coverage, units, observations
    )


def aggregate_patient_moments(
    table: PatientMomentTable,
    patient_multiplicities: Sequence[int] | None = None,
    *,
    min_independent_patients: int = 10,
    min_weighted_support: float = 0.90,
    locked_support: Sequence[DescriptorSupport] | None = None,
) -> MomentProfile:
    """Pool a cohort or bootstrap patient slots using identical conditional laws.

    For a bootstrap, pass the original profile's ``locked_support``. Original
        eligibility is retained; repeated sampled slots do not become new original
        patients, and a small count of distinct resampled originals is not a fresh
        inclusion test. Weighted support adequacy is checked in each resample and
        empty/undefined replicate moments remain undefined.
    """
    if min_independent_patients < 0 or not 0 <= min_weighted_support <= 1:
        raise ValueError("Invalid declared support thresholds")
    if patient_multiplicities is None:
        counts = np.ones(len(table.patient_ids), dtype=int)
    else:
        counts = np.asarray(patient_multiplicities)
        if (
            counts.ndim != 1
            or len(counts) != len(table.patient_ids)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)
        ):
            raise ValueError(
                "Patient multiplicities must be aligned nonnegative integers"
            )
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Patient sample is empty")
    if locked_support is not None and len(locked_support) != len(table.descriptor_ids):
        raise ValueError("Locked support must align with the descriptor universe")
    eligible_counts = counts @ table.eligible
    normalized = (
        counts[:, None] * table.eligible / np.maximum(eligible_counts, 1)[None, :]
    )
    anchor_rows = np.argmax(table.eligible, axis=0)
    columns = np.arange(len(table.descriptor_ids))
    mu_anchor = table.reference_mean[anchor_rows, columns]
    b_anchor = table.disagreement_mean[anchor_rows, columns]
    mu_offsets = table.reference_mean - mu_anchor
    b_offsets = table.disagreement_mean - b_anchor
    mu_offset = np.sum(normalized * mu_offsets, axis=0)
    b_offset = np.sum(normalized * b_offsets, axis=0)
    dx = mu_offsets - mu_offset
    db = b_offsets - b_offset
    v = np.sum(normalized * (table.reference_variance + dx * dx), axis=0)
    q = np.sum(normalized * (table.disagreement_variance + db * db), axis=0)
    a = np.sum(normalized * (table.reference_disagreement_covariance + dx * db), axis=0)
    coverage = counts @ table.original_support_coverage / total
    units = counts @ table.eligible_units
    observations = counts @ table.eligible_observations
    profile, supports = [], []
    for index in columns:
        reason = None
        if locked_support is not None:
            reason = locked_support[index].reason
        elif eligible_counts[index] < min_independent_patients:
            reason = "insufficient_independent_patients"
        if coverage[index] < min_weighted_support - 1e-12:
            reason = reason or "insufficient_original_weighted_support"
        if eligible_counts[index] == 0:
            moments = JointMoments(
                nan, nan, nan, nan, nan, nan, nan, "no_eligible_observations"
            )
            reason = reason or "no_eligible_observations"
        else:
            moments = _joint_from_components(
                float(mu_anchor[index] + mu_offset[index]),
                float(v[index]),
                float(b_anchor[index] + b_offset[index]),
                float(q[index]),
                float(a[index]),
            )
            if moments.reference_variance == 0:
                reason = reason or "constant_reference_variance"
            reason = reason or moments.ccc_reason
        if reason:
            moments = replace(moments, ccc=nan, ccc_reason=reason)
        profile.append(moments)
        supports.append(
            DescriptorSupport(
                int(eligible_counts[index]),
                int(units[index]),
                int(observations[index]),
                float(coverage[index]),
                reason,
            )
        )
    return MomentProfile(table.descriptor_ids, tuple(profile), tuple(supports))


def batch_joint_moments(
    reference: np.ndarray,
    alternate: np.ndarray,
    patient_ids: Sequence[str],
    unit_ids: Sequence[str],
    observation_ids: Sequence[str],
    *,
    eligible: np.ndarray,
    descriptor_ids: Sequence[str],
    min_independent_patients: int = 10,
    min_weighted_support: float = 0.90,
) -> MomentProfile:
    table = prepare_patient_moments(
        reference,
        alternate,
        patient_ids,
        unit_ids,
        observation_ids,
        eligible=eligible,
        descriptor_ids=descriptor_ids,
    )
    return aggregate_patient_moments(
        table,
        min_independent_patients=min_independent_patients,
        min_weighted_support=min_weighted_support,
    )
