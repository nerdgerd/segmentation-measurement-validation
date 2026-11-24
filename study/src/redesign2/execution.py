"""Explicit local input resolution for the published experiment entry points.

Historical manifests describe the bytes used for the recorded results. Their
source-file checks use the imported original snapshot; new runs separately stamp
the current code. Storage names never enter a random seed or a scientific choice.
"""

from pathlib import Path
import json
import os
import shutil
import tempfile

import pandas as pd

from .provenance import file_hash, resolve_input_path


def activate_inputs(workspace):
    workspace = Path(workspace).resolve()
    root = workspace.parent
    ledger = root / "private_inputs/path_map.json"
    imported = root / "private_inputs/IMPORT_RECORD.json"
    if not ledger.is_file() or not imported.is_file():
        raise FileNotFoundError(
            "Import the recorded inputs with tools/import_local_inputs.py first."
        )
    mapping = json.loads(ledger.read_text(encoding="utf-8"))
    original_root = Path(json.loads(imported.read_text(encoding="utf-8"))["source"])
    origin = original_root / "redesign2.0"
    for old, local in list(mapping.items()):
        original = Path(old)
        if original.is_relative_to(origin):
            alias = workspace / original.relative_to(origin)
            if not alias.exists():
                mapping[str(alias.resolve())] = local
        elif original.is_relative_to(original_root):
            alias = root / original.relative_to(original_root)
            if not alias.exists():
                mapping[str(alias.resolve())] = local
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=ledger.parent,
        prefix="runtime_paths-",
        suffix=".json",
        encoding="utf-8",
        newline="\n",
        delete=False,
    ) as handle:
        handle.write(json.dumps(mapping, indent=2) + "\n")
        runtime = handle.name
    os.environ["SMV_INPUT_MAP"] = runtime


def read_csv(path, *args, **kwargs):
    """Read one explicitly mapped input without changing pandas globally."""
    if isinstance(path, (str, Path)):
        path = resolve_input_path(path)
    return pd.read_csv(path, *args, **kwargs)


def copy_input(source, destination):
    return shutil.copyfile(resolve_input_path(source), destination)


def read_text(path, *args, **kwargs):
    return resolve_input_path(path).read_text(*args, **kwargs)


def storage_run_id(prefix, stage_name):
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("run_id must be a single nonempty path component")
    return f"{prefix}-{stage_name}"


def verify_frozen_input(path, expected):
    """Verify original locked bytes while allowing a declared source refactor."""
    path = Path(path).resolve()
    if path.exists() and file_hash(path) == expected:
        return path
    ledger = os.environ.get("SMV_INPUT_MAP")
    if not ledger:
        raise ValueError(f"Missing input ledger for locked input: {path}")
    workspace = Path(ledger).parent.parent / "study"
    record = workspace.parent / "private_inputs/IMPORT_RECORD.json"
    origin = (
        Path(json.loads(record.read_text(encoding="utf-8"))["source"]) / "redesign2.0"
    )
    original = origin / path.relative_to(workspace)
    archived = resolve_input_path(original)
    if not archived.is_file() or file_hash(archived) != expected:
        raise ValueError(f"Original locked input missing or changed: {original}")
    return archived
