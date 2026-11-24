"""A completed result cannot be replaced or silently changed."""

import json
from pathlib import Path

import pandas as pd
import pytest

from redesign2.provenance import RunWriter, read_protocol, verify_run


def test_atomic_run_and_tamper_detection(tmp_path, monkeypatch):
    workspace = Path(__file__).resolve().parents[1]
    cfg = read_protocol(workspace / "config/protocol.yaml")
    monkeypatch.setattr(
        "redesign2.provenance.code_state", lambda workspace: {"test": True}
    )
    source = tmp_path / "source.txt"
    source.write_text("input")
    writer = RunWriter(
        tmp_path, cfg, "test", "test-run", inputs=[source], estimand="test", role="test"
    )
    writer.write_table("values.csv", pd.DataFrame({"x": [1.25]}))
    assert not writer.final.exists()
    path = writer.finish()
    assert verify_run(path) == []
    with pytest.raises(FileExistsError):
        RunWriter(
            tmp_path, cfg, "test", "test-run", inputs=[], estimand="test", role="test"
        )
    (path / "values.csv").write_text("changed")
    assert any("Output missing or changed" in error for error in verify_run(path))
    source.write_text("new input")
    assert any("Input missing or changed" in error for error in verify_run(path))
