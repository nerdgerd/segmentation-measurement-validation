"""Guard physical phase, exact returned-mask calibration, and original-donor weights."""

import numpy as np
import pandas as pd
import pytest

from redesign2.adapters import diffuse
from redesign2.adapters.descriptors import descriptor_definedness
from redesign2.adapters.masks import MaskData, _resample_fixed_lattice, dice
from redesign2.adapters.prepare import require_mri_alignment, severity_donors


def test_calibration_exhaustion_reports_returned_mask(monkeypatch):
    def fake_apply(_prep, amplitude):
        mask = np.zeros((1, 1, 10), dtype=np.uint8)
        mask[..., : int(10 - amplitude)] = 3
        return mask

    monkeypatch.setattr(diffuse, "apply", fake_apply)
    ref = np.ones((1, 1, 10), bool)
    amp, achieved, iterations, converged = diffuse.calibrate(
        {}, ref, 0.5, max_iter=1, tol=1e-10, bracket=(0.0, 8.0)
    )
    assert amp == 6.0 and iterations == 1 and converged is False
    actual = fake_apply({}, amp) > 0
    assert achieved == 2 * actual.sum() / (ref.sum() + actual.sum())
    assert achieved != 2 * 6 / 16


def test_calibration_requires_explicit_rng():
    with pytest.raises(ValueError, match="declared"):
        diffuse.prepare(np.ones((3, 3, 3), np.uint8), rng=None)


def test_coherent_diffuse_partition_and_replay():
    seg = np.zeros((17, 17, 17), np.uint8)
    seg[4:13, 4:13, 4:13] = 2
    seg[6:11, 6:11, 6:11] = 1
    seg[8:10, 8:10, 8:10] = 3
    first = diffuse.prepare(seg, rng=np.random.default_rng(19))
    second = diffuse.prepare(seg, rng=np.random.default_rng(19))
    assert np.array_equal(diffuse.apply(first, 2), diffuse.apply(second, 2))
    assert set(np.unique(diffuse.apply(first, 2))) <= {0, 1, 2, 3}
    assert np.array_equal(diffuse.apply(first, 0), seg)


def test_dice_uses_global_origin_and_does_not_count_padding():
    a = MaskData(np.ones((2, 2, 2), bool), origin=(10, 10, 10))
    b = np.zeros((4, 4, 4), bool)
    b[1:3, 1:3, 1:3] = True
    assert dice(a, MaskData(b, origin=(9, 9, 9))) == 1
    assert dice(a, MaskData(b, origin=(20, 20, 20))) == 0


def test_ct_phase_is_fixed_when_native_crop_changes():
    original = np.zeros((35, 45, 45), bool)
    original[10:25, 13:32, 15:29] = True
    spacing = (2.5, 0.703125, 0.703125)
    full = _resample_fixed_lattice(original, spacing, (0, 0, 0), metadata={})
    crop = _resample_fixed_lattice(
        original[2:33, 3:41, 4:40], spacing, (2, 3, 4), metadata={}
    )
    assert dice(full, crop) == 1.0


def test_severity_mass_equal_patient_then_nodule_reader():
    rows = []
    for patient, units in (("p1", ["a", "b"]), ("p2", ["c"])):
        for unit in units:
            for reader in ("r1", "r2", "r3"):
                rows.append(
                    {
                        "cohort": "c",
                        "patient_id": patient,
                        "unit_id": unit,
                        "alternate_annotation_id": reader,
                        "observed_dice": 0.8,
                    }
                )
    donors = severity_donors(pd.DataFrame(rows))
    assert donors.groupby("patient_id").probability.sum().to_dict() == {
        "p1": 0.5,
        "p2": 0.5,
    }
    assert donors.groupby("unit_id").probability.sum().to_dict() == {
        "a": 0.25,
        "b": 0.25,
        "c": 0.5,
    }


def test_intercompartment_absence_is_not_numerical_failure():
    seg = np.zeros((10, 10, 10), np.uint8)
    seg[2:8, 2:8, 2:8] = 3
    assert (
        descriptor_definedness(np.nan, "ic__rim_completeness", seg, "MRI")
        == "structural_absence"
    )
    assert (
        descriptor_definedness(np.nan, "ET__solidity", seg, "MRI")
        == "numerical_undefined"
    )


def test_mri_physical_pair_alignment_rejects_origin_and_rotation_mismatch():
    def data(origin=(0, 0, 0), direction=None):
        geometry = {
            "size_xyz": [240, 240, 155],
            "spacing_xyz": [1, 1, 1],
            "origin_xyz": list(origin),
            "direction": list(np.eye(3).ravel() if direction is None else direction),
        }
        return MaskData(
            np.zeros((2, 2, 2), np.uint8), metadata={"image_geometry": geometry}
        )

    require_mri_alignment(data(), data(origin=(1e-7, 0, 0)))
    with pytest.raises(ValueError, match="physical lattice"):
        require_mri_alignment(data(), data(origin=(1e-5, 0, 0)))
    rotation = np.eye(3)
    rotation[0, 1] = 1e-7
    with pytest.raises(ValueError, match="physical lattice"):
        require_mri_alignment(data(), data(direction=rotation.ravel()))
