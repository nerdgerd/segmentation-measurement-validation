"""Intrinsic geometric descriptors of a binary 3-D mask.

Design rule, applied without exception: every *shape* feature returned here is
dimensionless. Rescaling a mask by any factor leaves it unchanged. Volume is
reported separately and is never treated as a shape feature. Consequence: a
shape feature that predicts survival cannot be volume in disguise -- which is
precisely the claim the project has to defend.

No intensity information is read. No atlas or anatomical reference is used.
Descriptors are invariant to translation and rotation.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes, mesh_surface_area

# Feature names, grouped. Order is fixed and is the pre-registered feature set.
SHAPE_FEATURES = [
    "sphericity",  # pi^(1/3)(6V)^(2/3)/A -- global compactness
    "solidity",  # V / V_convexhull -- how much of the hull is filled
    "surface_convexity",  # A_convexhull / A -- how much of the surface is convex
    "elongation",  # sqrt(l2/l1) of the inertia tensor
    "flatness",  # sqrt(l3/l1)
    "obb_fill",  # V / oriented-bounding-box volume
    "radial_cv",  # sd/mean of centroid->surface distance: lobulation
    "radial_p95_norm",  # 95th pct of that distance / equivalent-sphere radius
    "max_diameter_norm",  # Feret diameter / equivalent-sphere diameter
    "fractal_dim",  # box-counting dimension of the boundary
    "curvature_mean_norm",  # mean surface mean-curvature x R_eq
    "curvature_sd_norm",  # sd of surface mean-curvature x R_eq
    "thickness_cv",  # sd/mean of the interior distance transform
    "thickness_max_norm",  # max interior distance / equivalent-sphere radius
    "n_components",  # foci >= MIN_COMPONENT_VOXELS (multifocality)
    "largest_component_frac",  # V_largest / V_total (fragmentation)
    "cavity_frac",  # enclosed-void volume / filled volume
]
# Dropped before any outcome was examined: `genus_per_component`. Under a
# resolution-perturbation test it moved by up to 400% on thin enhancing shells,
# i.e. it counts voxelisation tunnels rather than anatomy.

MIN_COMPONENT_VOXELS = 64  # a focus smaller than this is segmentation speckle


# Surface-derived descriptors are measured on a size-normalised copy of the
# mask (equivalent-sphere radius forced to CANON_R voxels, anti-aliased by
# linear interpolation). Without this, a small tumour is sampled with fewer
# voxels than a large one and every surface feature inherits a covert volume
# dependence -- the exact confounder this project must rule out.
CANON_R = 28.0


MAX_CANON_VOXELS = 40e6  # ceiling on the resampled array (~160 MB as float32)


def _canonical(m):
    """Resample a binary mask to a fixed equivalent-sphere radius.

    The zoom factor comes from the volume-based equivalent radius but applies to
    the whole bounding box, so a thin, widely-spread shell (small volume, large
    bbox) would otherwise blow up: a 100^3 crop at 6x becomes 600^3. The zoom is
    additionally capped so the output array cannot exceed MAX_CANON_VOXELS.
    """
    r_eq = (3 * float(m.sum()) / (4 * np.pi)) ** (1 / 3)
    z = CANON_R / max(r_eq, 1e-6)
    if abs(z - 1.0) < 0.02:
        return m
    z = float(np.clip(z, 0.25, 6.0))
    z = min(z, (MAX_CANON_VOXELS / max(m.size, 1)) ** (1 / 3))
    if z <= 0.25:
        z = 0.25
    return ndi.zoom(m.astype(np.float32), z, order=1) > 0.5


def _crop(mask, pad=2):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]


def _surface_mesh(m):
    """Marching-cubes mesh of a binary volume, padded so the surface is closed."""
    p = np.pad(m.astype(np.float32), 1)
    verts, faces, _, _ = marching_cubes(p, level=0.5)
    return verts, faces


def _fractal_dimension(m):
    """Box-counting dimension of the mask boundary."""
    border = m ^ ndi.binary_erosion(m)
    counts, sizes = [], []
    for s in (1, 2, 3, 4, 6, 8, 12, 16):
        if min(border.shape) < 2 * s:
            break
        pad = [(0, (-d) % s) for d in border.shape]
        b = np.pad(border, pad)
        sh = b.reshape(b.shape[0] // s, s, b.shape[1] // s, s, b.shape[2] // s, s)
        n = int((sh.any(axis=(1, 3, 5))).sum())
        if n > 0:
            counts.append(n)
            sizes.append(s)
    if len(counts) < 3:
        return np.nan
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


def _curvature(m, verts):
    """Mean curvature at the surface, from the divergence of the unit gradient
    of a smoothed signed distance field. Returns (mean|H|, sd H)."""
    sdf = ndi.distance_transform_edt(m).astype(np.float32) - ndi.distance_transform_edt(
        ~m
    ).astype(np.float32)
    sdf = ndi.gaussian_filter(sdf, 1.5)
    g = np.gradient(sdf)
    n = np.sqrt(sum(x**2 for x in g)) + 1e-8
    div = sum(np.gradient(g[i] / n, axis=i) for i in range(3))
    H = 0.5 * div  # mean curvature
    # sample at mesh vertices (mesh lives on a 1-voxel padded grid)
    vi = np.rint(verts - 1).astype(int)
    ok = np.all((vi >= 0) & (vi < np.array(m.shape)), axis=1)
    vals = H[vi[ok, 0], vi[ok, 1], vi[ok, 2]]
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [1, 99])  # clip voxelisation spikes
    vals = np.clip(vals, lo, hi)
    return float(np.mean(np.abs(vals))), float(np.std(vals))


def compute(mask, min_voxels=50):
    """mask: boolean 3-D array on a 1 mm isotropic grid.

    Returns dict with volume_mm3 plus every entry of SHAPE_FEATURES.
    """
    out = {k: np.nan for k in SHAPE_FEATURES}
    out["volume_mm3"] = float(mask.sum())
    if mask.sum() < min_voxels:
        return out

    m = _crop(mask)
    V = float(m.sum())
    r_eq = (3 * V / (4 * np.pi)) ** (1 / 3)

    # --- components / fragmentation -------------------------------------
    lab, ncomp = ndi.label(m, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    out["n_components"] = float((sizes >= MIN_COMPONENT_VOXELS).sum())
    out["largest_component_frac"] = float(sizes.max() / V)

    # --- surface (measured on the size-normalised copy) -------------------
    _mc = _canonical(m)
    mc = _crop(_mc) if _mc.any() else m
    Vc = float(mc.sum())
    verts, faces = _surface_mesh(mc)
    A = float(mesh_surface_area(verts, faces))
    out["sphericity"] = float(np.pi ** (1 / 3) * (6 * Vc) ** (2 / 3) / A)

    # --- convex hull ----------------------------------------------------
    surf_pts = np.argwhere(m ^ ndi.binary_erosion(m)).astype(float)
    try:
        hull = ConvexHull(surf_pts)
        out["solidity"] = float(V / hull.volume)
        hp = surf_pts[hull.vertices]
        d = np.linalg.norm(hp[:, None, :] - hp[None, :, :], axis=-1)
        out["max_diameter_norm"] = float(d.max() / (2 * r_eq))
    except Exception:
        pass
    try:  # surface convexity also needs the normalised surface area
        hull_c = ConvexHull(np.argwhere(mc ^ ndi.binary_erosion(mc)).astype(float))
        out["surface_convexity"] = float(hull_c.area / A)
    except Exception:
        pass

    # --- inertia axes ---------------------------------------------------
    pts = np.argwhere(m).astype(float)
    c = pts.mean(0)
    cen = pts - c
    evals, evecs = np.linalg.eigh(np.cov(cen.T))
    evals = np.clip(evals[::-1], 1e-9, None)  # descending
    out["elongation"] = float(np.sqrt(evals[1] / evals[0]))
    out["flatness"] = float(np.sqrt(evals[2] / evals[0]))
    proj = cen @ evecs[:, ::-1]
    ext = proj.max(0) - proj.min(0) + 1.0
    out["obb_fill"] = float(V / np.prod(ext))

    # --- radial profile (lobulation) ------------------------------------
    r = np.linalg.norm(surf_pts - c, axis=1)
    out["radial_cv"] = float(r.std() / r.mean())
    out["radial_p95_norm"] = float(np.percentile(r, 95) / r_eq)

    # --- boundary roughness ---------------------------------------------
    out["fractal_dim"] = _fractal_dimension(mc)
    cm, cs = _curvature(mc, verts)
    out["curvature_mean_norm"] = cm * CANON_R if np.isfinite(cm) else np.nan
    out["curvature_sd_norm"] = cs * CANON_R if np.isfinite(cs) else np.nan

    # --- interior thickness ---------------------------------------------
    dt = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1, 1:-1][m]
    out["thickness_cv"] = float(dt.std() / dt.mean())
    out["thickness_max_norm"] = float(dt.max() / r_eq)

    # --- topology --------------------------------------------------------
    filled = np.stack([ndi.binary_fill_holes(s) for s in m])  # per-slice
    filled = ndi.binary_fill_holes(m) | filled
    out["cavity_frac"] = float((filled.sum() - V) / filled.sum())

    return out
