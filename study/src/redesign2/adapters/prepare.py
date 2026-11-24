"""Prepare one measurement unit in an explicitly authorized data role.

These helpers return values and masks; callers own versioned writes and role
separation. A construction process must receive only its separately materialized
reference products, never this all-contour adapter or the original mixed tables.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .descriptors import describe, descriptor_definedness
from .masks import dice, load_annotation_mask, reference_crop


def descriptor_manifest(repo_root, modality):
    path = Path(repo_root) / "study/audit/audited_descriptor_manifest.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    selected = frame.loc[frame.domain.eq(modality)].copy()
    if selected.empty or selected.feature_id.duplicated().any():
        raise ValueError("Missing or ambiguous audited descriptor identities")
    if selected[["family", "region", "units", "definition"]].isna().any().any():
        raise ValueError("Descriptor metadata contains an unresolved scientific field")
    return selected


def read_mri_features(repo_root, cohort, role):
    """Open one original MRI measurement role; no CCC/rank artifacts are read."""
    if cohort not in {"upenn", "brats_tcga"} or role not in {"reference", "alternate"}:
        raise ValueError(
            "A supported MRI cohort and explicit reference/alternate role are required"
        )
    tag = "tcga" if cohort == "brats_tcga" else cohort
    group, name = (
        ("reference_view", "reference_descriptor_rows.csv")
        if role == "reference"
        else ("evaluation_view", "observed_descriptor_rows.csv")
    )
    path = Path(repo_root) / "data/views" / group / tag / name
    frame = pd.read_csv(path, float_precision="round_trip")
    if frame.case.duplicated().any():
        raise ValueError("MRI measurement identity is not unique")
    ids = descriptor_manifest(repo_root, "MRI").feature_id.tolist()
    if not set(ids).issubset(frame.columns):
        raise ValueError("MRI table does not implement the audited descriptor universe")
    return frame[["case", *ids]].rename(columns={"case": "unit_id"})


def _flag(value, feature, data, modality):
    return descriptor_definedness(value, feature, data, modality)


def require_mri_alignment(reference, alternate, tolerance_mm=1e-6):
    """Identical array indices must identify the same physical lattice in a pair.

    Compare all eight image corners in physical coordinates so direction errors
    are judged in millimetres over the actual field of view, not angular units.
    """
    if tolerance_mm != 1e-6:
        raise ValueError("MRI header tolerance is a fixed technical audit rule")
    ga, gb = reference.metadata["image_geometry"], alternate.metadata["image_geometry"]
    if ga["size_xyz"] != gb["size_xyz"] or reference.mask.shape != alternate.mask.shape:
        raise ValueError("MRI source pair image dimensions differ")
    indices = np.array(
        list(itertools.product(*[(0, n - 1) for n in ga["size_xyz"]])), float
    )

    def physical(geometry):
        direction = np.asarray(geometry["direction"], float).reshape(3, 3)
        coordinates = indices * np.asarray(geometry["spacing_xyz"], float)
        return coordinates @ direction.T + np.asarray(geometry["origin_xyz"], float)

    distance = np.linalg.norm(physical(ga) - physical(gb), axis=1)
    if not np.isfinite(distance).all() or distance.max() > tolerance_mm:
        raise ValueError("MRI source pair physical lattice mismatch")


def prepare_unit(
    repo_root,
    contour_rows,
    reference_annotation_id,
    *,
    modality,
    feature_ids=None,
    reuse_mri_tables=True,
):
    """Return exact per-contour measurements and reference-to-alternate source Dice.

    This function is for isolated structural materialization/source preparation.
    CT always re-extracts on the new lattice. MRI may reuse the audited per-contour
    measurements after loading masks for severity and descriptor definedness flags.
    """
    rows = (
        contour_rows.to_dict("records")
        if isinstance(contour_rows, pd.DataFrame)
        else list(contour_rows)
    )
    if not rows or len({r["unit_id"] for r in rows}) != 1:
        raise ValueError("Prepare exactly one complete measurement unit")
    rows = [dict(row) for row in rows]
    for row in rows:
        row["stable_annotation_id"] = str(row["stable_annotation_id"])
    references = [
        r for r in rows if r["stable_annotation_id"] == str(reference_annotation_id)
    ]
    if len(references) != 1 or len(rows) < 2:
        raise ValueError(
            "Exactly one selected reference and at least one alternate are required"
        )
    ids = (
        list(feature_ids)
        if feature_ids is not None
        else descriptor_manifest(repo_root, modality).feature_id.tolist()
    )
    masks, features = {}, []
    for row in sorted(rows, key=lambda r: r["stable_annotation_id"]):
        identity = row["stable_annotation_id"]
        data = load_annotation_mask(repo_root, row)
        masks[identity] = data
        role = "reference" if identity == str(reference_annotation_id) else "alternate"
        if modality == "MRI" and reuse_mri_tables:
            table = read_mri_features(repo_root, row["cohort_id"], role)
            match = table.loc[table.unit_id.eq(row["unit_id"])]
            if len(match) != 1:
                raise ValueError("Missing MRI table measurement")
            values = match.iloc[0].to_dict()
        else:
            values = describe(data, modality)
        if not set(ids).issubset(values):
            raise ValueError("Extraction did not return all descriptor identities")
        for feature in ids:
            value = float(values[feature])
            features.append(
                {
                    "cohort": row["cohort_id"],
                    "patient_id": row["patient_id"],
                    "nodule_id": row.get("nodule_id", ""),
                    "unit_id": row["unit_id"],
                    "annotation_id": identity,
                    "role": role,
                    "descriptor_id": feature,
                    "value": value,
                    "definedness": _flag(value, feature, data, modality),
                    "source_path": row["source_path"],
                    "source_sha256": row["source_sha256"],
                    "geometry_rule": data.metadata["grid_rule"],
                }
            )
    reference = masks[str(reference_annotation_id)]
    if modality == "MRI":
        for identity, data in masks.items():
            if identity != str(reference_annotation_id):
                require_mri_alignment(reference, data)
    base = references[0]
    severity = [
        {
            "cohort": base["cohort_id"],
            "patient_id": base["patient_id"],
            "nodule_id": base.get("nodule_id", ""),
            "unit_id": base["unit_id"],
            "reference_annotation_id": str(reference_annotation_id),
            "alternate_annotation_id": identity,
            "severity_region": "WT" if modality == "MRI" else "single",
            "observed_dice": dice(reference, data),
            "grid_rule": reference.metadata["grid_rule"],
        }
        for identity, data in masks.items()
        if identity != str(reference_annotation_id)
    ]
    cropped = reference_crop(reference)
    return {
        "features": pd.DataFrame(features),
        "severity": pd.DataFrame(severity),
        "reference_mask": cropped,
        "reference_mask_array_sha256": hashlib.sha256(
            np.ascontiguousarray(cropped.mask).tobytes()
        ).hexdigest(),
        "mask_metadata": {identity: data.metadata for identity, data in masks.items()},
    }


def severity_donors(severity_rows):
    """Each patient has equal mass, then units and remaining readers split that mass."""
    frame = severity_rows.copy()
    keys = ["patient_id", "unit_id", "alternate_annotation_id"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Source severity must contain unique observed pairs")
    if frame.cohort.nunique() != 1:
        raise ValueError("Severity distribution must contain exactly one source cohort")
    if (
        not np.isfinite(frame.observed_dice).all()
        or not frame.observed_dice.between(0, 1).all()
    ):
        raise ValueError(
            "Missing or invalid source severity must be resolved explicitly"
        )
    frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    patients = frame.patient_id.nunique()
    units = frame.groupby("patient_id").unit_id.transform("nunique")
    readers = frame.groupby(
        ["patient_id", "unit_id"]
    ).alternate_annotation_id.transform("nunique")
    frame["probability"] = 1.0 / (patients * units * readers)
    if not np.isclose(frame.probability.sum(), 1.0):
        raise ValueError("Invalid hierarchical severity distribution")
    return frame


def sample_donor(donors, rng):
    """A discrete original donor supplies the severity and retained donor identity."""
    index = int(rng.choice(len(donors), p=donors.probability.to_numpy(float)))
    return donors.iloc[index].to_dict()
