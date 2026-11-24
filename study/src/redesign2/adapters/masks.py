"""Each contour is prepared without another contour's geometry.

CT uses the native image's axis origin as a fixed 1 mm lattice anchor. An
annotation's own padded native crop determines its computational extent, while
the output phase is fixed by image metadata. This explicitly corrects the old
all-reader-union/zero-origin resampling and requires new CT descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from ..provenance import resolve_input_path
import configparser
import hashlib
import json
import warnings

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


@dataclass(frozen=True)
class MaskData:
    mask: np.ndarray
    origin: tuple[int, int, int] = (0, 0, 0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    metadata: dict = field(default_factory=dict)


def _path(root, path):
    value = Path(path)
    return resolve_input_path(value if value.is_absolute() else Path(root) / value)


@lru_cache(maxsize=64)
def _validate_source(path, size, modified_ns, expected):
    # Size and modification time invalidate the cache; repeated NSCLC reader
    # extraction otherwise rehashes the same large SEG five times.
    with Path(path).open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Audited source content changed: {path}")


def _require_original_source(repo_root, row):
    source = _path(repo_root, row["source_path"]).resolve()
    stat = source.stat()
    _validate_source(str(source), stat.st_size, stat.st_mtime_ns, row["source_sha256"])


def reference_crop(data, margin=12):
    positions = np.argwhere(data.mask > 0)
    if not len(positions):
        raise ValueError("Cannot generate from an empty reference")
    lo = np.maximum(positions.min(0) - margin, 0)
    hi = np.minimum(positions.max(0) + margin + 1, data.mask.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return MaskData(
        np.ascontiguousarray(data.mask[slices]),
        tuple((np.asarray(data.origin) + lo).tolist()),
        data.spacing,
        data.metadata,
    )


def _resample_fixed_lattice(mask, spacing_zyx, native_start_zyx, *, metadata):
    """The crop origin is retained in native image-axis millimetres before interpolation."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("Structurally empty annotation")
    spacing = np.asarray(spacing_zyx, float)
    native_start = np.asarray(native_start_zyx, int)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Invalid native spacing")
    distance = (
        ndi.distance_transform_edt(mask, sampling=spacing)
        - ndi.distance_transform_edt(~mask, sampling=spacing)
    ).astype(np.float32)
    physical_start = native_start * spacing
    working_start = np.floor(physical_start).astype(int)
    working_end = np.ceil(physical_start + np.asarray(mask.shape) * spacing).astype(int)
    src = sitk.GetImageFromArray(distance)
    src.SetSpacing(tuple(spacing[::-1]))
    src.SetOrigin(tuple(physical_start[::-1]))
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((1.0, 1.0, 1.0))
    resampler.SetOutputOrigin(tuple(working_start[::-1].astype(float)))
    resampler.SetSize([int(v) for v in (working_end - working_start)[::-1]])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(distance.min()))
    out = sitk.GetArrayFromImage(resampler.Execute(src)) >= 0.0
    provenance = {
        **metadata,
        "grid_rule": "native_image_axis_origin_anchored_1mm",
        "native_spacing_zyx": spacing.tolist(),
        "native_crop_start_zyx": native_start.tolist(),
        "working_origin_zyx": working_start.tolist(),
        "resample": "physical_signed_distance_linear_threshold_ge_zero",
    }
    return MaskData(out, tuple(working_start.tolist()), metadata=provenance)


def load_annotation_mask(repo_root, row):
    """Read exactly the annotated contour; row comes from audited contour metadata."""
    row = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    _require_original_source(repo_root, row)
    cohort = row["cohort_id"]
    if cohort in {"upenn", "brats_tcga"}:
        image = sitk.ReadImage(str(_path(repo_root, row["source_path"])))
        if not np.allclose(image.GetSpacing(), (1, 1, 1), atol=1e-12, rtol=0):
            raise ValueError("MRI native mask is not 1 mm isotropic")
        raw = sitk.GetArrayFromImage(image)
        if not set(np.unique(raw)).issubset({0, 1, 2, 4}):
            raise ValueError("Unexpected MRI source labels")
        mask = raw.astype(np.uint8)
        mask[mask == 4] = 3
        geometry = {
            "size_xyz": list(image.GetSize()),
            "spacing_xyz": list(image.GetSpacing()),
            "origin_xyz": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
        }
        return MaskData(
            mask,
            metadata={
                "source_path": row["source_path"],
                "source_sha256": row["source_sha256"],
                "grid_rule": "native_1mm_MRI",
                "label_mapping": "0,1,2 unchanged;4->3",
                "image_geometry": geometry,
            },
        )
    if cohort == "nsclc":
        return _nsclc_mask(repo_root, row)
    if cohort == "lidc":
        return _lidc_mask(row)
    raise ValueError(f"Unknown cohort: {cohort}")


def _nsclc_mask(repo_root, row):
    import pydicom

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(_path(repo_root, row["ct_dir"])))
    if not files:
        raise FileNotFoundError(row["ct_dir"])
    reader.SetFileNames(files)
    image = reader.Execute()
    ds = pydicom.dcmread(_path(repo_root, row["source_path"]))
    number = int(float(row["segment_number"]))
    exact_id = f"SEG:{ds.SOPInstanceUID}#segment:{number}"
    if exact_id != row["stable_annotation_id"]:
        raise ValueError("SEG annotation identity changed")
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    native = np.zeros(tuple(reversed(image.GetSize())), dtype=bool)
    origin, spacing = np.asarray(image.GetOrigin()), np.asarray(image.GetSpacing())
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    for index, group in enumerate(ds.PerFrameFunctionalGroupsSequence):
        if (
            int(group.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
            != number
        ):
            continue
        position = np.asarray(
            group.PlanePositionSequence[0].ImagePositionPatient, float
        )
        voxel = np.linalg.solve(direction * spacing, position - origin)
        k = int(round(voxel[2]))
        if (
            np.max(np.abs(voxel - np.round(voxel))) > 0.01
            or not 0 <= k < native.shape[0]
        ):
            raise ValueError("SEG frame does not map to the audited CT grid")
        native[k] |= frames[index].astype(bool)
    locations = np.argwhere(native)
    if not len(locations):
        raise ValueError("Selected NSCLC annotation is empty")
    pad = np.ceil(20.0 / spacing[::-1]).astype(int)
    lo = np.maximum(locations.min(0) - pad, 0)
    hi = np.minimum(locations.max(0) + pad + 1, native.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    geometry = {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": spacing.tolist(),
        "origin_xyz": origin.tolist(),
        "direction": direction.ravel().tolist(),
    }
    geometry_hash = hashlib.sha256(
        json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _resample_fixed_lattice(
        native[slices],
        spacing[::-1],
        lo,
        metadata={
            "source_path": row["source_path"],
            "source_sha256": row["source_sha256"],
            "annotation_id": exact_id,
            "ct_dir": row["ct_dir"],
            "image_origin_xyz": origin.tolist(),
            "image_direction": direction.ravel().tolist(),
            "ct_image_geometry": geometry,
            "ct_image_geometry_sha256": geometry_hash,
            "native_pad_mm": 20.0,
        },
    )


def _lidc_module():
    """Compatibility aliases are process-local; the annotation DB connection is read-only."""
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser
    for name, target in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pylidc
    return pylidc


def _lidc_mask(row):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    pl = _lidc_module()
    db = resolve_input_path(row["source_path"]).resolve()
    engine = create_engine("sqlite:///file:" + db.as_posix() + "?mode=ro&uri=true")
    with Session(engine) as session:
        annotation = (
            session.query(pl.Annotation)
            .filter(pl.Annotation.id == int(row["stable_annotation_id"]))
            .one()
        )
        if annotation.scan.patient_id != row["patient_id"] or int(
            annotation.scan.id
        ) != int(float(row["scan_id"])):
            raise ValueError("LIDC annotation/patient/scan identity changed")
        ps, st = (
            float(annotation.scan.pixel_spacing),
            float(annotation.scan.slice_thickness),
        )
        padding = [(int(np.ceil(12.0 / ps)),) * 2] * 2 + [
            (int(np.ceil(12.0 / st)),) * 2
        ]
        bbox = annotation.bbox(pad=padding)
        native = annotation.boolean_mask(pad=padding)
        native_start = [part.start for part in bbox]
        # pylidc row,col,slice -> extractor slice,row,col, with the same physical axes.
        data = _resample_fixed_lattice(
            np.transpose(native, (2, 0, 1)),
            (st, ps, ps),
            (native_start[2], native_start[0], native_start[1]),
            metadata={
                "source_path": str(db),
                "source_sha256": row["source_sha256"],
                "annotation_id": row["stable_annotation_id"],
                "scan_id": int(float(row["scan_id"])),
                "native_pad_mm": 12.0,
                "native_axis_order": "pylidc_row_col_slice",
                "cluster_recomputed": False,
            },
        )
    engine.dispose()
    return data


def dice(a: MaskData, b: MaskData, *, region=None):
    """Align two cropped masks by their global lattice offsets, without changing either."""
    if a.spacing != b.spacing:
        raise ValueError("Masks use incompatible physical grids")
    am = np.isin(a.mask, region) if region is not None else a.mask > 0
    bm = np.isin(b.mask, region) if region is not None else b.mask > 0
    denominator = int(am.sum()) + int(bm.sum())
    if denominator == 0:
        return 1.0
    astart, bstart = np.asarray(a.origin), np.asarray(b.origin)
    lo = np.maximum(astart, bstart)
    hi = np.minimum(astart + am.shape, bstart + bm.shape)
    if np.any(hi <= lo):
        return 0.0
    sa = tuple(slice(int(x), int(y)) for x, y in zip(lo - astart, hi - astart))
    sb = tuple(slice(int(x), int(y)) for x, y in zip(lo - bstart, hi - bstart))
    return float(2 * np.count_nonzero(am[sa] & bm[sb]) / denominator)
