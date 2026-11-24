"""The publication interface must work without the original machine or patient data."""

from pathlib import Path
import importlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from redesign2.execution import read_csv, verify_frozen_input
from redesign2.moments import measurement_difference_ratio
from redesign2.provenance import file_hash, verify_run
from study.experiments.catalogue import EXPERIMENTS
from study.experiments.measurement_differences import replay
from study.experiments.reference_results import (
    manuscript_summary,
    verify_threshold_points,
)

ROOT = Path(__file__).resolve().parents[2]


def test_experiment_imports_do_not_open_inputs_or_run_computations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import attempted a study read or run")

    from redesign2.provenance import RunWriter

    original_read_text = Path.read_text

    def study_read(path, *args, **kwargs):
        if path.resolve().is_relative_to(ROOT):
            forbidden()
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(Path, "read_text", study_read)
    monkeypatch.setattr(RunWriter, "__init__", forbidden)
    for item in EXPERIMENTS:
        module = importlib.import_module("study.experiments." + item.module)
        assert callable(module.run)


def test_cli_summary_works_from_an_unrelated_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/reproduce.py"), "summary"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(result.stdout)
    assert len(rows) == 4
    assert sorted(row["falsely_kept_at_095"] for row in rows) == [18, 21, 92, 93]


def test_replay_covers_every_recorded_cutoff_and_rmse_feature():
    # Two MRI directions x two analyses: 5 thresholds x 2 methods each, and
    # 4 x 192 features less the one ineligible in held-out UPENN-to-TCGA.
    assert verify_threshold_points(ROOT / "study/reference") == 40
    rows = replay(ROOT / "study/reference")
    assert len(rows.groupby(["source", "target", "scope"])) == 4
    assert np.isfinite(rows.recomputed_log2_rmse_ratio).sum() == 767
    assert len(manuscript_summary(ROOT / "study/reference")) == 4


@pytest.mark.parametrize(
    "generated,original,expected,stratum",
    [
        (1.0, 4.0, -1.0, "finite"),
        (4.0, 1.0, 1.0, "finite"),
        (4.0, 4.0, 0.0, "finite"),
        (0.0, 1.0, -np.inf, "synthetic_only_zero"),
        (1.0, 0.0, np.inf, "observed_only_zero"),
        (0.0, 0.0, np.nan, "both_zero"),
        (-1.0, 1.0, np.nan, "undefined"),
        (np.nan, 1.0, np.nan, "undefined"),
        (np.inf, 1.0, np.nan, "undefined"),
    ],
)
def test_extracted_ratio_keeps_zero_and_invalid_cases(
    generated, original, expected, stratum
):
    value, classification = measurement_difference_ratio(generated, original)
    assert classification == stratum
    assert (np.isnan(value) and np.isnan(expected)) or value == expected


def test_mapped_reader_does_not_modify_pandas(tmp_path, monkeypatch):
    missing = tmp_path / "original.csv"
    actual = tmp_path / "relocated.csv"
    actual.write_text("value\n3\n", encoding="utf-8")
    ledger = tmp_path / "mapping.json"
    ledger.write_text(
        json.dumps({str(missing.resolve()): str(actual)}), encoding="utf-8"
    )
    monkeypatch.setenv("SMV_INPUT_MAP", str(ledger))
    original_function = pd.read_csv
    assert read_csv(missing).value.tolist() == [3]
    assert pd.read_csv is original_function


def test_original_code_lock_never_masks_the_current_code_hash(tmp_path, monkeypatch):
    private = tmp_path / "private_inputs"
    private.mkdir()
    workspace = tmp_path / "study"
    current = workspace / "src/algorithm.py"
    current.parent.mkdir(parents=True)
    current.write_text("current source", encoding="utf-8")
    archived = private / "original.py"
    archived.write_text("original source", encoding="utf-8")
    origin = tmp_path / "unavailable_origin"
    (private / "IMPORT_RECORD.json").write_text(
        json.dumps({"source": str(origin)}), encoding="utf-8"
    )
    ledger = private / "runtime_paths.json"
    key = origin / "redesign2.0/src/algorithm.py"
    ledger.write_text(json.dumps({str(key.resolve()): str(archived)}), encoding="utf-8")
    monkeypatch.setenv("SMV_INPUT_MAP", str(ledger))
    expected = file_hash(archived)
    assert verify_frozen_input(current, expected) == archived
    assert file_hash(current) != expected
    archived.write_text("changed original", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_frozen_input(current, expected)


def test_relocated_run_source_is_verified_and_tampering_still_fails(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    run.mkdir()
    archived = tmp_path / "archived_source.py"
    archived.write_text("original code", encoding="utf-8")
    recorded_path = run / "snapshot_source.py"
    manifest = {"outputs": {"snapshot_source.py": file_hash(archived)}, "inputs": {}}
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ledger = tmp_path / "mapping.json"
    ledger.write_text(
        json.dumps({str(recorded_path.resolve()): str(archived)}), encoding="utf-8"
    )
    monkeypatch.setenv("SMV_INPUT_MAP", str(ledger))
    assert verify_run(run) == []
    archived.write_text("tampered code", encoding="utf-8")
    assert verify_run(run) == ["Output missing or changed: snapshot_source.py"]
