"""One partition supplies every MRI measurement; shape definitions remain frozen.

The only copied volume convention is the existing 1e-6 voxel denominator in
volume ratios. It is preserved as part of descriptor identity, not used in RMSE
ratios or the redesigned estimators. No extraction runner/configuration is imported.
"""

import numpy as np

from . import features_extended as fe
from . import shape_features as sf

REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "NETC": (1,)}


def descriptor_definedness(value, feature, maskdata, modality):
    """Separate anatomical absence and the primitive's documented support guards.

    The intercompartment requirements below are copied from the explicit guards
    in features_extended.intercompartment_features, not inferred from missing rates.
    Other unresolved numerical failures remain numerical_undefined.
    """
    if np.isfinite(value):
        return "defined"
    mask = np.asarray(getattr(maskdata, "mask", maskdata))
    if feature.startswith("ic__") and modality == "MRI":
        counts = {
            region: int(np.count_nonzero(np.isin(mask, labels)))
            for region, labels in REGIONS.items()
        }
        counts["SNFH"] = int(np.count_nonzero(mask == 2))
        stem = feature.removeprefix("ic__")
        requirements = {"TC": 100}
        if stem in {
            "rim_completeness",
            "rim_thickness_mean_norm",
            "rim_thickness_cv",
            "necrosis_eccentricity",
            "necrosis_buried_frac",
            "et_interface_frac",
        }:
            requirements.update(NETC=50, ET=50)
        elif stem == "et_surface_share":
            requirements["ET"] = 50
        elif stem.startswith("edema_") or stem == "core_offset_in_wt":
            requirements["SNFH"] = 100
        elif stem == "compartment_nesting":
            requirements["NETC"] = 50
        if any(counts[region] == 0 for region in requirements):
            return "structural_absence"
        if any(
            counts[region] < threshold for region, threshold in requirements.items()
        ):
            return "below_intercompartment_voxel_support"
        if stem == "edema_anisotropy":
            return "insufficient_angular_support"
    elif "__" in feature:
        region = feature.split("__", 1)[0]
        labels = REGIONS.get(region)
        count = (
            int(np.count_nonzero(mask > 0))
            if modality == "CT"
            else int(np.count_nonzero(np.isin(mask, labels)))
            if labels
            else None
        )
        if count == 0:
            return "structural_absence"
        if count is not None and count < 50:
            return "below_50_voxel_shape_support"
    return "numerical_undefined"


def describe(maskdata, modality):
    mask = np.asarray(getattr(maskdata, "mask", maskdata))
    spacing = np.asarray(getattr(maskdata, "spacing", (1.0, 1.0, 1.0)), float)
    if not np.array_equal(spacing, np.ones(3)):
        raise ValueError(
            "Descriptor extraction requires the audited 1 mm isotropic grid"
        )
    if modality not in {"MRI", "CT"}:
        raise ValueError(f"Unknown modality: {modality}")
    if modality == "CT":
        return _single(mask > 0)
    if not set(np.unique(mask)).issubset({0, 1, 2, 3}):
        raise ValueError("MRI masks require the harmonized 0/1/2/3 partition")
    volume = {
        region: float(np.isin(mask, labs).sum()) for region, labs in REGIONS.items()
    }
    volume["SNFH"] = float((mask == 2).sum())
    result = {f"vol_{region}_cm3": value / 1000.0 for region, value in volume.items()}
    for numerator, denominator in (
        ("SNFH", "ET"),
        ("NETC", "ET"),
        ("ET", "TC"),
        ("TC", "WT"),
        ("NETC", "TC"),
    ):
        result[f"volratio_{numerator}_over_{denominator}"] = volume[numerator] / (
            volume[denominator] + 1e-6
        )
    for region, labs in REGIONS.items():
        result.update(_shape(np.isin(mask, labs), region))
    result.update(
        {
            f"ic__{key}": value
            for key, value in fe.intercompartment_features(mask).items()
        }
    )
    return result


def _shape(mask, region):
    result = {
        f"{region}__{key}": value
        for key, value in sf.compute(mask, min_voxels=50).items()
        if key != "volume_mm3"
    }
    result.update(
        {
            f"{region}__{key}": value
            for key, value in fe.region_features(mask, spharm_on_raw=True).items()
        }
    )
    return result


def _single(mask):
    result = _shape(mask, "WT")
    result["vol_WT_cm3"] = float(mask.sum()) / 1000.0
    return result
