"""Content identities and atomic new runs prevent silent replacement of evidence.

The protocol's canonical JSON representation is hashed independently of YAML layout.
Seeds include the master seed, study namespace, named stream, and explicit task keys.
Run IDs name storage only and never influence scientific randomness.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def resolve_input_path(value) -> Path:
    """Resolve an archived absolute input through an explicit local path ledger."""
    path = Path(value)
    ledger = os.environ.get("SMV_INPUT_MAP")
    if ledger:
        mapping = json.loads(Path(ledger).read_text(encoding="utf-8"))
        key = str(path.resolve())
        if key in mapping:
            return Path(mapping[key])
    return path


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with resolve_input_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_protocol(path: Path | str) -> dict:
    with resolve_input_path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def subseed(cfg: dict, stream: str, *keys) -> int:
    spec = cfg["randomness"]
    if stream not in spec["streams"]:
        raise ValueError(f"Undeclared random stream: {stream}")
    key = [spec["master_seed"], spec["namespace"], spec["streams"][stream], *keys]
    return int.from_bytes(hashlib.sha256(canonical_bytes(key)).digest()[:16], "big")


def rng(cfg: dict, stream: str, *keys) -> np.random.Generator:
    return np.random.default_rng(subseed(cfg, stream, *keys))


def inside(root: Path | str, path: Path | str) -> Path:
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Expected a child path of {root}: {path}")
    return path


def environment() -> dict:
    packages = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "PyYAML",
        "matplotlib",
        "pytest",
        "nibabel",
        "SimpleITK",
        "scikit-image",
        "pyarrow",
        "pydicom",
        "pylidc",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def code_state(workspace: Path | str) -> dict:
    workspace = Path(workspace).resolve()
    repo = workspace.parent

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo)

    patch = git("diff", "--binary", "HEAD")
    sources = {
        str(p.relative_to(workspace)).replace("\\", "/"): file_hash(p)
        for folder in ["src", "tests", "experiments", "figures"]
        for p in sorted((workspace / folder).rglob("*.py"))
    }
    sources["../pyproject.toml"] = file_hash(repo / "pyproject.toml")
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "tracked_dirty_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "status": git("status", "--porcelain").decode().splitlines(),
        "workspace_source_hashes": sources,
        "workspace_source_sha256": content_hash(sources),
    }


class RunWriter:
    """Stage output becomes visible only after a complete manifest is written.

    Incomplete temporary directories are retained as evidence of interrupted work.
    Existing completed runs are never overwritten. All writes stay under workspace.
    """

    def __init__(
        self,
        workspace,
        cfg,
        stage,
        run_id,
        *,
        inputs,
        estimand,
        role,
        seeds=None,
        status="complete",
    ):
        self.workspace = Path(workspace).resolve()
        self.cfg = cfg
        self.stage = stage
        parent = inside(self.workspace, self.workspace / "runs" / content_hash(cfg))
        parent.mkdir(parents=True, exist_ok=True)
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a single nonempty path component")
        self.final = inside(parent, parent / run_id)
        if self.final.exists():
            raise FileExistsError(self.final)
        self.temp = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=parent))
        self.manifest = {
            "stage": stage,
            "run_id": run_id,
            "status": status,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": content_hash(cfg),
            "code": code_state(self.workspace),
            "environment": environment(),
            "estimand": estimand,
            "role": role,
            "inputs": {str(Path(p).resolve()): file_hash(p) for p in inputs},
            "seeds": seeds or {},
            "reference_rule": cfg.get("reference_selection"),
            "weighting": cfg.get("weighting"),
            "outputs": {},
        }

    def write_text(self, name: str, value: str):
        path = inside(self.temp, self.temp / name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(value)

    def write_json(self, name: str, value):
        self.write_text(
            name,
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )

    def write_table(self, name, frame):
        frame = frame.copy()
        frame["protocol_sha256"] = content_hash(self.cfg)
        frame["stage"] = self.stage
        frame["run_id"] = self.manifest["run_id"]
        frame["declared_estimand"] = self.manifest["estimand"]
        self.write_text(name, frame.to_csv(index=False, lineterminator="\n"))

    def finish(self, extra=None) -> Path:
        if self.final.exists():
            raise FileExistsError(self.final)
        self.manifest["extra"] = extra or {}
        self.manifest["outputs"] = {
            str(p.relative_to(self.temp)).replace("\\", "/"): file_hash(p)
            for p in sorted(self.temp.rglob("*"))
            if p.is_file()
        }
        self.write_json("run_manifest.json", self.manifest)
        os.rename(self.temp, self.final)
        return self.final


def verify_run(path: Path | str, *, verify_inputs=True) -> list[str]:
    path = resolve_input_path(path).resolve()
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for name, expected in manifest["outputs"].items():
        file = resolve_input_path(inside(path, path / name))
        if not file.is_file() or file_hash(file) != expected:
            errors.append(f"Output missing or changed: {name}")
    actual = {
        str(p.relative_to(path)).replace("\\", "/")
        for p in path.rglob("*")
        if p.is_file()
    }
    extra = actual - set(manifest["outputs"]) - {"run_manifest.json"}
    errors.extend(f"Unmanifested output: {p}" for p in sorted(extra))
    if verify_inputs:
        for name, expected in manifest["inputs"].items():
            if not resolve_input_path(name).is_file() or file_hash(name) != expected:
                errors.append(f"Input missing or changed: {name}")
    return errors
