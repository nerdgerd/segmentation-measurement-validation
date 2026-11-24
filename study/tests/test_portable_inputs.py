"""A clean checkout must verify archived inputs without their original paths."""

import json

from redesign2.provenance import file_hash, read_protocol, resolve_input_path


def test_explicit_mapping_verifies_the_archived_bytes(tmp_path, monkeypatch):
    original = tmp_path / "unavailable_machine/input.csv"
    relocated = tmp_path / "inputs/input.csv"
    relocated.parent.mkdir()
    relocated.write_text("feature,value\nf,2\n", encoding="utf-8")
    ledger = tmp_path / "map.json"
    ledger.write_text(
        json.dumps({str(original.resolve()): str(relocated)}), encoding="utf-8"
    )
    monkeypatch.setenv("SMV_INPUT_MAP", str(ledger))
    assert resolve_input_path(original) == relocated
    assert file_hash(original) == file_hash(relocated)


def test_configuration_contents_and_seed_namespace_are_preserved(tmp_path, monkeypatch):
    original = tmp_path / "old_machine/protocol.yaml"
    actual = tmp_path / "protocol.yaml"
    actual.write_text(
        "randomness:\n  namespace: original_namespace\n  master_seed: 20260921\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "map.json"
    ledger.write_text(
        json.dumps({str(original.resolve()): str(actual)}), encoding="utf-8"
    )
    monkeypatch.setenv("SMV_INPUT_MAP", str(ledger))
    assert read_protocol(original) == read_protocol(actual)


def test_unmapped_paths_are_not_guessed(tmp_path, monkeypatch):
    ledger = tmp_path / "map.json"
    ledger.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SMV_INPUT_MAP", str(ledger))
    missing = tmp_path / "not_in_the_record.csv"
    assert resolve_input_path(missing) == missing
