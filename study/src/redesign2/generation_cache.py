"""Cache complete generation jobs by scientific inputs and numerical source hashes.

Each committed cache contains actual 20-draw value/calibration/definedness files.
It is reusable only after all recorded table hashes and schemas are verified.
Incomplete directories and corrupted completed entries are never overwritten.
Target alternate measurements are not inputs to this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .provenance import (
    code_state,
    content_hash,
    environment,
    file_hash,
    inside,
    subseed,
)


NUMERICAL_SOURCES = (
    "src/redesign2/adapters/diffuse.py",
    "src/redesign2/adapters/descriptors.py",
    "src/redesign2/adapters/features_extended.py",
    "src/redesign2/adapters/shape_features.py",
    "src/redesign2/adapters/masks.py",
    "src/redesign2/generate.py",
    "src/redesign2/provenance.py",
)
TABLE_NAMES = ("values", "calibration", "definedness")


class CacheCorruptionError(ValueError):
    pass


def _request(args, workspace):
    (
        cfg,
        source,
        target,
        unit,
        patient,
        path,
        expected_hash,
        donors,
        feature_ids,
        start,
        end,
    ) = args
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        raise ValueError("Draw intervals require nonnegative integer start < end")
    if len(feature_ids) != len(set(feature_ids)) or not feature_ids:
        raise ValueError("Distinct descriptor identities are required")
    reference = Path(path).resolve()
    if not reference.is_file() or file_hash(reference) != expected_hash:
        raise ValueError(f"Reference mask hash changed or input missing: {unit}")
    numerical_sources = {
        name: file_hash(workspace / name) for name in NUMERICAL_SOURCES
    }
    software = environment()
    request = {
        "format_version": 1,
        "protocol_sha256": content_hash(cfg),
        "reference_sha256": expected_hash,
        "source_donor_records_sha256": content_hash(donors),
        "source": source,
        "target": target,
        "unit_id": unit,
        "patient_id": patient,
        "descriptor_ids": list(feature_ids),
        "draw_start": start,
        "draw_end": end,
        "storage_batch_size": 20,
        "numerical_sources": numerical_sources,
        "software_environment": software,
    }
    return request, reference


def _validate_frames(frames, request):
    if not isinstance(frames, (tuple, list)) or len(frames) != 3:
        raise ValueError("Generation must return values, calibration, and definedness")
    start, end = request["draw_start"], request["draw_end"]
    expected = list(range(start, end))
    for name, frame in zip(TABLE_NAMES, frames, strict=True):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError(f"Invalid {name} frame")
        if not {"draw_id", "batch", "unit_id", "patient_id"}.issubset(frame.columns):
            raise ValueError(f"Missing identity columns in {name}")
        if frame.draw_id.tolist() != expected:
            raise ValueError(
                f"{name} must contain every requested draw exactly once in order"
            )
        if not np.array_equal(frame.batch.to_numpy(), frame.draw_id.to_numpy() // 20):
            raise ValueError(f"Incorrect absolute storage batch in {name}")
        if (
            not frame.unit_id.eq(request["unit_id"]).all()
            or not frame.patient_id.eq(request["patient_id"]).all()
        ):
            raise ValueError(f"Wrong patient/unit identity in {name}")
    for frame in [frames[0], frames[2]]:
        if not set(request["descriptor_ids"]).issubset(frame.columns):
            raise ValueError(
                "Values and definedness must contain the declared descriptor identities"
            )
        if "cohort" not in frame or not frame.cohort.eq(request["target"]).all():
            raise ValueError("Values or definedness name a different target cohort")
    if not {"source", "target"}.issubset(frames[1].columns):
        raise ValueError("Calibration must name its source and target")
    if (
        not frames[1].source.eq(request["source"]).all()
        or not frames[1].target.eq(request["target"]).all()
    ):
        raise ValueError("Calibration names a different source-target direction")


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _read_table(path, schema):
    dtypes = schema["dtypes"]
    na_values = {
        column: [""] for column, dtype in dtypes.items() if dtype.startswith("float")
    }
    frame = pd.read_csv(
        path,
        float_precision="round_trip",
        keep_default_na=False,
        dtype=dtypes,
        na_values=na_values,
    )
    if frame.columns.tolist() != schema["columns"] or len(frame) != schema["rows"]:
        raise CacheCorruptionError(f"Cached table schema changed: {path}")
    if {column: str(dtype) for column, dtype in frame.dtypes.items()} != dtypes:
        raise CacheCorruptionError(f"Cached table dtypes changed: {path}")
    return frame


def _read_committed(directory, request):
    manifest_path = directory / "cache_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["request"] != request or manifest["cache_key"] != content_hash(
            request
        ):
            raise CacheCorruptionError(f"Cache manifest identity mismatch: {directory}")
        expected_paths = set(manifest["outputs"]) | {"cache_manifest.json"}
        actual_paths = {
            str(p.relative_to(directory)).replace("\\", "/")
            for p in directory.rglob("*")
            if p.is_file()
        }
        if actual_paths != expected_paths:
            raise CacheCorruptionError(
                f"Missing or unmanifested cache files: {directory}"
            )
        for relative, record in manifest["outputs"].items():
            path = inside(directory, directory / relative)
            if file_hash(path) != record["sha256"]:
                raise CacheCorruptionError(f"Cached table hash changed: {path}")
        frames = []
        for name in TABLE_NAMES:
            paths = manifest["tables"][name]
            fragments = [
                _read_table(
                    inside(directory, directory / relative),
                    manifest["outputs"][relative],
                )
                for relative in paths
            ]
            frames.append(pd.concat(fragments, ignore_index=True))
        _validate_frames(frames, request)
        return tuple(frames)
    except CacheCorruptionError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CacheCorruptionError(
            f"Cannot verify cached generation at {directory}: {error}"
        ) from error


def cached_generate_job(args, workspace):
    """Wrap the lazy generator with a verified immutable, content-addressed cache."""
    workspace = Path(workspace).resolve()
    request, reference = _request(args, workspace)
    cache_key = content_hash(request)
    parent = inside(workspace, workspace / "cache" / "generation")
    parent.mkdir(parents=True, exist_ok=True)
    final = inside(parent, parent / cache_key)
    if final.exists():
        return _read_committed(final, request)

    from .generate import _generate_job

    frames = _generate_job(args)
    _validate_frames(frames, request)
    if file_hash(reference) != request["reference_sha256"]:
        raise ValueError("Reference changed during generation; output cannot be cached")
    if {name: file_hash(workspace / name) for name in NUMERICAL_SOURCES} != request[
        "numerical_sources"
    ]:
        raise ValueError(
            "Numerical source changed during generation; output cannot be cached"
        )
    temp = Path(tempfile.mkdtemp(prefix=f".{cache_key}-", dir=parent))
    cfg, source, target, unit, patient = args[:5]
    manifest = {
        "cache_key": cache_key,
        "request": request,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "declared_estimand": "single_draw_from_finite_search_Diffuse_generator",
        "information_role": "source_donors_and_target_reference_only",
        "target_alternates_read": False,
        "code": code_state(workspace),
        "input_files": {
            str(reference): request["reference_sha256"],
            **{
                str(workspace / name): digest
                for name, digest in request["numerical_sources"].items()
            },
        },
        "source_donor_records": args[7],
        "seeds": [
            {
                "keys": [source, target, patient, unit, draw],
                "field_seed": str(
                    subseed(
                        cfg, "perturbation_fields", source, target, patient, unit, draw
                    )
                ),
                "donor_seed": str(
                    subseed(cfg, "severity_donors", source, target, patient, unit, draw)
                ),
            }
            for draw in range(request["draw_start"], request["draw_end"])
        ],
        "outputs": {},
        "tables": {name: [] for name in TABLE_NAMES},
    }
    for name, frame in zip(TABLE_NAMES, frames, strict=True):
        for batch in sorted(frame.batch.unique()):
            fragment = frame.loc[frame.batch.eq(batch)].reset_index(drop=True)
            relative = f"batch_{batch:03d}/{name}.csv"
            path = inside(temp, temp / relative)
            _write_text(path, fragment.to_csv(index=False, lineterminator="\n"))
            manifest["outputs"][relative] = {
                "sha256": file_hash(path),
                "rows": len(fragment),
                "columns": fragment.columns.tolist(),
                "dtypes": {
                    column: str(dtype) for column, dtype in fragment.dtypes.items()
                },
            }
            manifest["tables"][name].append(relative)
    _write_text(
        temp / "cache_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    # A racing writer may have committed the same scientific job. Verify its entry;
    # retain this temporary directory as an uncommitted record instead of deleting it.
    try:
        os.rename(temp, final)
    except FileExistsError:
        return _read_committed(final, request)
    return _read_committed(final, request)
