import json

import pytest

from redesign2.commitments import (
    verify_construction_commitment,
    write_construction_commitment,
)


def test_changed_selection_or_input_invalidates_commitment(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("source-only")
    score = tmp_path / "scores.json"
    score.write_text("permitted scores")
    path = tmp_path / "commitment.json"
    selections = {
        key: ["a"] for key in ["source_ccc", "synthetic_ccc", "error_transport"]
    }
    write_construction_commitment(
        path,
        protocol_hash="test",
        direction="a_to_b",
        scope="full",
        universe=["a", "b", "c", "d"],
        selections=selections,
        inputs=[source],
        artifacts=[score],
    )
    assert (
        verify_construction_commitment(path, protocol_hash="test")["direction"]
        == "a_to_b"
    )
    source.write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_construction_commitment(path, protocol_hash="test")
    source.write_text("source-only")
    data = json.loads(path.read_text())
    data["selections"]["source_ccc"] = ["b"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash changed"):
        verify_construction_commitment(path, protocol_hash="test")
