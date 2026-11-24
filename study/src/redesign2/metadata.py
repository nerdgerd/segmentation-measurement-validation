"""Reference and discovery choices depend on stable identifiers alone."""

from collections import defaultdict
import hashlib

from .provenance import canonical_bytes


def metadata_digest(*key):
    return hashlib.sha256(canonical_bytes(list(key))).hexdigest()


def select_ct_references(namespace, cohort, annotations):
    """Rows contain only patient_id, nodule_id and stable_annotation_id strings."""
    groups = defaultdict(list)
    seen = set()
    allowed = {"patient_id", "nodule_id", "stable_annotation_id"}
    for row in annotations:
        if set(row) != allowed or any(
            not isinstance(v, str) or not v for v in row.values()
        ):
            raise ValueError(
                "Reference selection accepts three nonempty metadata identifiers only"
            )
        patient, nodule, annotation = (
            row[k] for k in ("patient_id", "nodule_id", "stable_annotation_id")
        )
        identity = patient, nodule, annotation
        if identity in seen:
            raise ValueError(f"Duplicate annotation identity: {identity}")
        seen.add(identity)
        groups[(patient, nodule)].append(
            (metadata_digest(namespace, cohort, *identity), annotation)
        )
    selected = []
    for (patient, nodule), candidates in sorted(groups.items()):
        if len(candidates) < 2:
            raise ValueError("Reference units need an eligible alternate")
        digest, annotation = min(candidates)
        selected.append(
            {
                "cohort": cohort,
                "patient_id": patient,
                "nodule_id": nodule,
                "reference_annotation_id": annotation,
                "reference_digest": digest,
                "annotation_count": len(candidates),
            }
        )
    return selected


def patient_split(namespace, cohort, patients):
    patients = list(patients)
    if any(not isinstance(p, str) or not p for p in patients):
        raise ValueError("Patient identifiers must be nonempty strings")
    ordered = sorted(
        set(patients), key=lambda p: (metadata_digest(namespace, cohort, p), p)
    )
    cut = len(ordered) // 2
    return [
        {
            "cohort": cohort,
            "patient_id": patient,
            "assignment": "discovery" if i < cut else "audit",
            "split_digest": metadata_digest(namespace, cohort, patient),
        }
        for i, patient in enumerate(ordered)
    ]
