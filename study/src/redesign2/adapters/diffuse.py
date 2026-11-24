"""Explicit-argument fork of frozen Diffuse, without cfg import side effects.

The field, partition, bracket, tolerance and bisection are retained. Exhausted
bisection now reports Dice at the returned amplitude, correcting the frozen
amplitude/Dice mismatch. No failed attempt is replaced or omitted here.
"""

import numpy as np
from scipy import ndimage as ndi

TUMOUR = (1, 2, 3)


def prepare(seg, *, rng, sigma_mm=6.0):
    if rng is None:
        raise ValueError("A declared seeded generator is required")
    seg = np.asarray(seg)
    if seg.ndim != 3 or not set(np.unique(seg)).issubset({0, 1, 2, 3}):
        raise ValueError("Expected a harmonized three-dimensional partition")
    result = {"shape": seg.shape, "labels": [], "sdf": [], "field": [], "bias": 0.0}
    for label in TUMOUR:
        mask = seg == label
        if not mask.any():
            continue
        noise = ndi.gaussian_filter(
            rng.standard_normal(seg.shape).astype(np.float32), sigma_mm
        )
        noise = noise / (noise.std() + 1e-8)
        sdf = ndi.distance_transform_edt(mask).astype(
            np.float32
        ) - ndi.distance_transform_edt(~mask).astype(np.float32)
        result["labels"].append(label)
        result["field"].append(noise)
        result["sdf"].append(sdf)
    if result["labels"]:
        result["field"] = np.stack(result["field"])
        result["sdf"] = np.stack(result["sdf"])
    return result


def apply(prep, amplitude):
    result = np.zeros(prep["shape"], dtype=np.uint8)
    if not prep["labels"]:
        return result
    stack = prep["sdf"] + amplitude * (prep["field"] + prep["bias"])
    best, inside = stack.argmax(0), stack.max(0) > 0
    for index, label in enumerate(prep["labels"]):
        result[inside & (best == index)] = label
    return result


def _dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else float(2 * np.count_nonzero(a & b) / total)


def calibrate(
    prep,
    ref_mask,
    target,
    *,
    region=TUMOUR,
    tol=0.005,
    max_iter=24,
    bracket=(0.05, 16.0),
):
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("Requested Dice must be finite and in [0,1]")
    if max_iter < 1 or tol <= 0 or not 0 <= bracket[0] < bracket[1]:
        raise ValueError("Invalid calibration configuration")
    lo, hi = map(float, bracket)

    def dice_at(amplitude):
        return _dice(ref_mask, np.isin(apply(prep, amplitude), region))

    d_lo, d_hi = dice_at(lo), dice_at(hi)
    if d_lo < target:
        return lo, d_lo, 0, False
    if d_hi > target:
        return hi, d_hi, 0, False
    for index in range(max_iter):
        mid = 0.5 * (lo + hi)
        achieved = dice_at(mid)
        if abs(achieved - target) <= tol:
            return mid, achieved, index + 1, True
        if achieved > target:
            lo = mid
        else:
            hi = mid
    returned_amplitude = 0.5 * (lo + hi)
    return returned_amplitude, dice_at(returned_amplitude), max_iter, False
