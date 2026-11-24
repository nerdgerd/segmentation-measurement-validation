"""Evaluation opens only after all directional construction artifacts are sealed."""

import json
from pathlib import Path

from .provenance import content_hash, file_hash, inside, resolve_input_path


def write_construction_commitment(
    path, *, protocol_hash, direction, scope, universe, selections, inputs, artifacts
):
    path = Path(path)
    payload = {
        "protocol_sha256": protocol_hash,
        "direction": direction,
        "scope": scope,
        "universe": sorted(universe),
        "universe_sha256": content_hash(sorted(universe)),
        "selections": {key: list(value) for key, value in selections.items()},
        "inputs": {str(Path(p).resolve()): file_hash(p) for p in inputs},
        "artifacts": {str(Path(p).resolve()): file_hash(p) for p in artifacts},
    }
    if len(set(universe)) != len(universe):
        raise ValueError("Duplicate descriptor identities in construction universe")
    if set(selections) != {"source_ccc", "synthetic_ccc", "error_transport"}:
        raise ValueError("All three procedures must be committed together")
    if not universe:
        raise ValueError("Empty construction universe cannot be committed as evaluable")
    k = max(1, len(universe) // 4)
    for selection in selections.values():
        if (
            len(selection) != k
            or len(set(selection)) != k
            or set(selection) - set(universe)
        ):
            raise ValueError("Selection does not match the locked universe and budget")
    payload["commitment_sha256"] = content_hash(payload)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def verify_construction_commitment(path, *, protocol_hash):
    data = json.loads(resolve_input_path(path).read_text(encoding="utf-8"))
    recorded = data.pop("commitment_sha256")
    if content_hash(data) != recorded or data["protocol_sha256"] != protocol_hash:
        raise ValueError("Construction commitment or protocol hash changed")
    for category in ["inputs", "artifacts"]:
        for name, expected in data[category].items():
            if not resolve_input_path(name).is_file() or file_hash(name) != expected:
                raise ValueError(f"Committed {category} missing or changed: {name}")
    return data


def require_all_constructions(cfg, commitments):
    expected = {
        (f"{d['source']}_to_{d['target']}", scope)
        for d in cfg["directions"]
        for scope in ["full", "discovery"]
    }
    verified = [
        verify_construction_commitment(path, protocol_hash=content_hash(cfg))
        for path in commitments
    ]
    found = [(entry["direction"], entry["scope"]) for entry in verified]
    if len(found) != len(set(found)) or set(found) != expected:
        raise PermissionError(
            "Every full and discovery construction must be committed before evaluation"
        )
    return verified
