"""Role-specific file capabilities keep alternate tables out of construction.

This is a scientific access contract, not an operating-system security boundary.
Construction receives an allowlist containing only separately materialized permitted
files. It must never receive a global all-cohort table or import an evaluation loader.
"""

from dataclasses import dataclass
from pathlib import Path

from .provenance import file_hash


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str
    direction: str
    role: str
    scope: str


class ConstructionInputs:
    def __init__(self, direction, scope, assets):
        if scope not in {"full", "discovery"}:
            raise ValueError("Construction scope must be full or discovery")
        self.direction, self.scope = direction, scope
        self.assets = {}
        self.access_record = []
        for name, asset in assets.items():
            permitted = asset.role in {
                "source_product",
                "target_reference",
                "synthetic",
            }
            appropriate_scope = asset.scope == scope or (
                asset.role == "source_product" and asset.scope == "source"
            )
            if asset.direction != direction or not permitted or not appropriate_scope:
                raise PermissionError(f"Forbidden construction asset: {name}")
            self.assets[name] = asset

    def read_bytes(self, name):
        if name not in self.assets:
            raise PermissionError(f"Asset not in construction allowlist: {name}")
        asset = self.assets[name]
        if file_hash(asset.path) != asset.sha256:
            raise ValueError(f"Input hash changed: {name}")
        self.access_record.append(
            {
                "name": name,
                "sha256": asset.sha256,
                "direction": self.direction,
                "scope": self.scope,
            }
        )
        return asset.path.read_bytes()


CACHE_KEYS = (
    "reference_hash",
    "generator_hash",
    "severity_distribution_hash",
    "coherent_draw_rule",
    "extraction_hash",
    "seed_namespace",
    "draw_semantics",
    "weighting",
    "severity_donor_ids",
)


def require_compatible_cache(expected, actual):
    missing = [k for k in CACHE_KEYS if k not in expected or k not in actual]
    different = [
        k
        for k in CACHE_KEYS
        if k in expected and k in actual and expected[k] != actual[k]
    ]
    if missing or different:
        raise ValueError(
            f"Incompatible cache; missing={missing}; different={different}"
        )
