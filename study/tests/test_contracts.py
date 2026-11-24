"""Changing identities or exposing evaluation inputs must fail before ranking."""

import ast
import json
from pathlib import Path

import pytest

from redesign2.access import (
    Asset,
    CACHE_KEYS,
    ConstructionInputs,
    require_compatible_cache,
)
from redesign2.metadata import patient_split, select_ct_references
from redesign2.provenance import (
    canonical_bytes,
    content_hash,
    file_hash,
    inside,
    read_protocol,
    subseed,
)


def test_reference_selection_is_order_independent_and_geometry_rejected():
    rows = [
        {"patient_id": "p", "nodule_id": "n", "stable_annotation_id": a}
        for a in ["c", "a", "b"]
    ]
    assert select_ct_references("r", "ct", rows) == select_ct_references(
        "r", "ct", rows[::-1]
    )
    with pytest.raises(ValueError, match="metadata"):
        select_ct_references("r", "ct", [{**rows[0], "volume": 50}])


def test_split_keeps_patient_together_and_odd_remainder_in_audit():
    rows = patient_split("s", "ct", ["p1", "p2", "p2", "p3", "p4", "p5"])
    assert len(rows) == 5
    assert sum(r["assignment"] == "discovery" for r in rows) == 2
    assert rows == patient_split("s", "ct", ["p5", "p4", "p3", "p2", "p1"])


def test_construction_denies_reverse_direction_audit_reference_and_alternate(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("{}")

    def asset(role="target_reference", direction="a_to_b", scope="discovery"):
        return Asset(path, file_hash(path), direction, role, scope)

    for bad in [
        asset(role="target_alternate"),
        asset(direction="b_to_a"),
        asset(scope="audit"),
        asset(scope="full"),
    ]:
        with pytest.raises(PermissionError):
            ConstructionInputs("a_to_b", "discovery", {"bad": bad})
    view = ConstructionInputs("a_to_b", "discovery", {"reference": asset()})
    assert view.read_bytes("reference") == b"{}"
    with pytest.raises(PermissionError):
        view.read_bytes("target_alternate")
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        view.read_bytes("reference")


def test_cache_rejects_old_reference_and_regionwise_draws():
    expected = {key: "locked" for key in CACHE_KEYS}
    require_compatible_cache(expected, expected.copy())
    for key in [
        "reference_hash",
        "coherent_draw_rule",
        "severity_donor_ids",
        "weighting",
    ]:
        with pytest.raises(ValueError, match="Incompatible"):
            require_compatible_cache(expected, {**expected, key: "legacy"})
    with pytest.raises(ValueError):
        require_compatible_cache({}, {})


def test_seed_canonicalization_and_namespace_separation():
    workspace = Path(__file__).resolve().parents[1]
    cfg = read_protocol(workspace / "config/protocol.yaml")
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert subseed(cfg, "smoke", "x", 1) == subseed(cfg, "smoke", "x", 1)
    assert subseed(cfg, "smoke", "x", 1) != subseed(cfg, "mechanism_final", "x", 1)
    assert subseed(cfg, "smoke", "x", 1) != subseed(cfg, "smoke", "x", "1")
    with pytest.raises(ValueError):
        subseed(cfg, "unregistered")
    with pytest.raises(ValueError):
        canonical_bytes(float("nan"))


def test_write_path_cannot_escape_workspace(tmp_path):
    with pytest.raises(ValueError):
        inside(tmp_path, tmp_path / ".." / "outside")
    with pytest.raises(ValueError):
        inside(tmp_path, tmp_path)


def test_construction_import_graph_has_no_evaluation_or_legacy_runner():
    root = Path(__file__).resolve().parents[1] / "src/redesign2"
    queue, visited = ["access", "policies", "transport"], set()
    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        path = root / f"{name}.py"
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.level and node.module:
                    queue.append(node.module.split(".")[0])
            for module in modules:
                assert not any(
                    part in module.split(".")
                    for part in ["evaluate", "external_validation", "analysis", "views"]
                )
