"""Stage only the inputs used by retained experiments; keep patient assets local.

Same-filesystem hard links avoid duplicating large immutable draw tables. Existing
files are checked rather than overwritten. The explicit map preserves original
absolute input identities without rewriting frozen manifests or configurations.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Original GBM2 repository root"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy large inputs instead of linking immutable files",
    )
    args = parser.parse_args()
    origin = args.source.resolve()
    source_study = origin / "redesign2.0"
    if not (source_study / "config/protocol.lock.yaml").is_file():
        raise ValueError("Source is not the recorded GBM2 study checkout")
    private = ROOT / "private_inputs"
    private.mkdir(exist_ok=True)
    pending = deque()
    for name in (
        "subjects_metadata.csv",
        "contours_metadata.csv",
        "audited_descriptor_manifest.csv",
        "asset_facts.json",
    ):
        pending.append(source_study / "audit" / name)
    for folder, names in {
        "critical_revision": (
            "AMENDMENT_LOCK.json",
            "descriptor_definition_audit.csv",
            "analysis_package.json",
            "aggregation_package.json",
            "ZERO_POINT_DIAGNOSTICS_RULE.md",
        ),
        "media_revision": ("analysis_package.json",),
    }.items():
        for name in names:
            path = source_study / folder / name
            if path.is_file():
                pending.append(path)
    for name in (
        "MEDIA_REVISION_LOCK.json",
        "MEDIA_REVISION_AMENDMENT.md",
        "V16_SNAPSHOT.json",
        "MEDIA_REVISION_EXTREME_TRACE_ADDENDUM.md",
    ):
        path = source_study / name
        if path.is_file():
            pending.append(path)
    for folder in (source_study / "runs").iterdir():
        if not folder.is_dir():
            continue
        for run in folder.iterdir():
            if not run.is_dir() or not (run / "run_manifest.json").is_file():
                continue
            # Stage interfaces needed by final analyses; exclude pilots and manuscript packages.
            keep = run.name.startswith(
                (
                    "prepared-",
                    "generate-",
                    "construct-",
                    "evaluate-",
                    "finite-",
                    "coverage-",
                    "population-",
                    "simulate-",
                )
            ) or run.name in {
                "study-report-unitprecision-v2",
                "metadata-20260915",
                "existing-data-v1",
                "matched-finite-v1",
                "threshold-decision-fidelity-v1",
                "existing-data-reanalysis-v1",
                "current-aggregation-v1",
                "bootstrap-reasons-v1",
                "case-mix-strata-v1",
                "retained-volume-ratio-trace-v1",
            }
            if keep and "pilot" not in run.name:
                pending.append(run / "run_manifest.json")
    mapping = {}
    seen = set()
    unavailable = []
    byte_count = 0
    while pending:
        source = pending.popleft().resolve()
        if source in seen:
            continue
        seen.add(source)
        if not source.is_file():
            unavailable.append(str(source))
            continue
        if source.is_relative_to(source_study):
            relative = source.relative_to(source_study)
            # Exported operational source is kept separate from historical source identities.
            target = (
                STUDY / relative
                if relative.parts[0]
                in {"runs", "critical_revision", "media_revision", "audit"}
                and source.suffix != ".py"
                else private / "historical_study" / relative
            )
        elif source.is_relative_to(origin):
            target = private / "historical_repository" / source.relative_to(origin)
        else:
            # Imaging is acquired separately; record a verified original location without copying it.
            mapping[str(source)] = str(source)
            continue
        target = target.resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError(f"Destination escaped the clean repository: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != sha(source):
                raise ValueError(f"Existing imported input changed: {target}")
        elif not args.copy and source.stat().st_size > 1_000_000:
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        else:
            shutil.copyfile(source, target)
        mapping[str(source)] = str(target)
        byte_count += source.stat().st_size
        if source.suffix == ".json":
            try:
                record = json.loads(source.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError):
                continue
            if not isinstance(record, dict):
                continue
            if source.name == "run_manifest.json":
                for name in record.get("outputs", {}):
                    pending.append(source.parent / name)
                for name in record.get("inputs", {}):
                    pending.append(Path(name))
            for name in record.get("input_hashes", {}):
                pending.append(source_study / name)
            if "run" in record and isinstance(record["run"], str):
                pending.append(source_study / record["run"] / "run_manifest.json")
    # Exact MRI per-contour extraction views are an input interface, not a legacy analysis.
    for source in (origin / "data/views").glob("*/*/*descriptor_rows.csv"):
        target = ROOT / source.relative_to(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        mapping[str(source.resolve())] = str(target.resolve())
    # Store the audited legacy model primitive required only by the original protocol hash.
    legacy = origin / "src/v2/perturb_models.py"
    if legacy.is_file():
        target = private / "historical_repository/src/v2/perturb_models.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy, target)
        mapping[str(legacy)] = str(target)
    (private / "path_map.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (private / "IMPORT_RECORD.json").write_text(
        json.dumps(
            {
                "source": str(origin),
                "files": len(mapping),
                "logical_bytes": byte_count,
                "unavailable": unavailable,
                "patient_data_tracked": False,
                "large_input_storage": "copies"
                if args.copy
                else "immutable hard links where supported",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "files": len(mapping),
                "logical_bytes": byte_count,
                "unavailable_count": len(unavailable),
                "record": str(private / "IMPORT_RECORD.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
