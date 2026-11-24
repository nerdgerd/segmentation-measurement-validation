from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from redesign2 import generate, generation_cache as cache
from redesign2.provenance import file_hash


@pytest.fixture
def setup(tmp_path, monkeypatch):
    workspace = tmp_path / "redesign_workspace"
    workspace.mkdir()
    for name in cache.NUMERICAL_SOURCES:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# independently hashed numerical fixture\n", encoding="utf-8")
    reference = workspace / "reference.npz"
    reference.write_bytes(b"reference-mask-bytes-v1")
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "protocol.yaml").read_text()
    )
    donors = [
        {
            "patient_id": "source_001",
            "unit_id": "source_unit",
            "observed_dice": 0.8,
            "probability": 1.0,
            "alternate_annotation_id": "a2",
            "reference_annotation_id": "a1",
        }
    ]
    args = (
        cfg,
        "upenn",
        "brats_tcga",
        "unit_001",
        "patient_001",
        str(reference),
        file_hash(reference),
        donors,
        ["test_descriptor"],
        0,
        100,
    )
    state = {"calls": 0, "generated": None}

    def fake_generator(job):
        state["calls"] += 1
        _, source, target, unit, patient, _, reference_hash, _, _, start, end = job
        draws = np.arange(start, end)
        shared = {
            "cohort": target,
            "patient_id": patient,
            "unit_id": unit,
            "draw_id": draws,
            "batch": draws // 20,
        }
        values = pd.DataFrame(
            {
                **shared,
                "coherent_draw_id": [f"id_{i:03d}" for i in draws],
                "test_descriptor": np.linspace(
                    0.12345678901234566, 0.9876543210123456, len(draws)
                ),
            }
        )
        values.loc[0, "test_descriptor"] = np.nan
        calibration = pd.DataFrame(
            {
                "source": source,
                "target": target,
                "patient_id": patient,
                "unit_id": unit,
                "draw_id": draws,
                "batch": draws // 20,
                "reference_sha256": reference_hash,
                "field_seed": [str(2**120 + int(i)) for i in draws],
                "source_annotation_id": "0001",
                "converged": True,
                "failure_reason": "",
                "seconds": 0.12345678901234566,
            }
        )
        flags = pd.DataFrame(
            {
                **shared,
                "test_descriptor": ["structural_absence"]
                + ["defined"] * (len(draws) - 1),
            }
        )
        state["generated"] = (values, calibration, flags)
        return state["generated"]

    monkeypatch.setattr(generate, "_generate_job", fake_generator)
    monkeypatch.setattr(
        cache,
        "environment",
        lambda: {"python": "fixture", "packages": {"numpy": "fixture"}},
    )
    monkeypatch.setattr(
        cache,
        "code_state",
        lambda root: {
            "commit": "fixture_commit",
            "tracked_dirty_patch_sha256": "fixture_dirty",
        },
    )
    return workspace, args, state


def committed(workspace):
    return [
        p
        for p in (workspace / "cache" / "generation").iterdir()
        if not p.name.startswith(".")
    ]


def test_cache_round_trips_values_strings_flags_and_real_twenty_draw_files(setup):
    workspace, args, state = setup
    first = cache.cached_generate_job(args, workspace)
    second = cache.cached_generate_job(args, workspace)
    assert state["calls"] == 1
    for actual, again, expected in zip(first, second, state["generated"], strict=True):
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(actual, again)
    (directory,) = committed(workspace)
    files = sorted(directory.glob("batch_*/*.csv"))
    assert len(files) == 15
    for file in files:
        assert len(pd.read_csv(file)) == 20
    manifest = json.loads((directory / "cache_manifest.json").read_text())
    assert manifest["target_alternates_read"] is False
    assert len(manifest["seeds"]) == 100
    assert manifest["source_donor_records"] == args[7]
    assert manifest["request"]["reference_sha256"] == args[6]
    assert len(manifest["request"]["numerical_sources"]) == len(cache.NUMERICAL_SOURCES)


def test_corrupted_cache_is_rejected_and_never_overwritten(setup):
    workspace, args, state = setup
    cache.cached_generate_job(args, workspace)
    (directory,) = committed(workspace)
    file = directory / "batch_000" / "values.csv"
    file.write_text(file.read_text() + "corruption\n")
    changed = file.read_bytes()
    with pytest.raises(cache.CacheCorruptionError, match="hash changed"):
        cache.cached_generate_job(args, workspace)
    assert state["calls"] == 1
    assert file.read_bytes() == changed


def test_protocol_reference_donors_and_numerical_source_each_invalidate_cache(setup):
    workspace, args, state = setup
    cache.cached_generate_job(args, workspace)
    changed = list(args)
    changed[0] = deepcopy(args[0])
    changed[0]["synthetic"]["draw_count_initial"] = 200
    cache.cached_generate_job(tuple(changed), workspace)
    reference = Path(args[5])
    reference.write_bytes(b"reference-mask-bytes-v2")
    changed = list(args)
    changed[6] = file_hash(reference)
    cache.cached_generate_job(tuple(changed), workspace)
    changed[7] = deepcopy(args[7])
    changed[7][0]["observed_dice"] = 0.7
    cache.cached_generate_job(tuple(changed), workspace)
    numerical = workspace / cache.NUMERICAL_SOURCES[0]
    numerical.write_text("# changed numerical implementation\n")
    cache.cached_generate_job(tuple(changed), workspace)
    assert state["calls"] == 5
    assert len(committed(workspace)) == 5


def test_reference_integrity_checked_before_cache_hit(setup):
    workspace, args, state = setup
    cache.cached_generate_job(args, workspace)
    Path(args[5]).write_bytes(b"changed-reference-with-stale-expected-hash")
    with pytest.raises(ValueError, match="Reference mask hash changed"):
        cache.cached_generate_job(args, workspace)
    assert state["calls"] == 1


def test_different_draw_interval_preserves_absolute_batch_ids(setup):
    workspace, args, state = setup
    job = (*args[:-2], 100, 200)
    frames = cache.cached_generate_job(job, workspace)
    assert frames[0].draw_id.tolist() == list(range(100, 200))
    (directory,) = committed(workspace)
    assert sorted(p.name for p in directory.glob("batch_*")) == [
        f"batch_{i:03d}" for i in range(5, 10)
    ]


def test_missing_or_unmanifested_files_are_not_accepted(setup):
    workspace, args, state = setup
    cache.cached_generate_job(args, workspace)
    (directory,) = committed(workspace)
    (directory / "unmanifested.csv").write_text("untracked\n")
    with pytest.raises(cache.CacheCorruptionError, match="unmanifested"):
        cache.cached_generate_job(args, workspace)
    assert state["calls"] == 1
