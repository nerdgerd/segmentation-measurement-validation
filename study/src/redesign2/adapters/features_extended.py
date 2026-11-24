"""Expanded geometric feature battery -- six families beyond the intrinsic core.

Design rules carried over unchanged from `shape_features`:
  * every descriptor is dimensionless, so nothing here can be volume in disguise;
  * no voxel intensity is read (the anatomical reference is a fixed population
    template, not a per-patient image measurement);
  * families 3-6 are computed on the size-normalised canonical mask, which makes
    them scale-invariant by construction and bounds the compute cost.

Families
  loc__    location relative to brain anatomy (ventricles/SVZ, midline, cortex)
  ic__     inter-compartment geometry (rim, necrosis eccentricity, oedema shell)
  gran__   granulometry -- volume surviving opening at radii scaled to the object
  curv__   surface curvature classes (pushing vs infiltrative margin)
  skel__   medial-axis / tendril structure
  spec__   spherical-harmonic power spectrum of the radial function
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.special import sph_harm_y
from skimage.morphology import skeletonize

from .shape_features import _canonical, _crop, CANON_R

# ---- family definitions ---------------------------------------------------
GRAN_RADII = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]  # as a fraction of R_eq
CURV_CLASSES = ["cup", "rut", "saddle", "ridge", "cap"]
SPHARM_L = 10
SVZ_CONTACT_MM = 5.0  # standard subventricular-zone contact band

GRAN_FEATURES = [f"gran_r{int(r * 100):02d}" for r in GRAN_RADII] + ["gran_half_radius"]
CURV_FEATURES = [f"curv_{c}_frac" for c in CURV_CLASSES] + [
    "curvedness_mean",
    "curvedness_sd",
]
# Lee's 3-D thinning returns a medial *surface* for blob-like objects, not a
# curve, so neighbour-count "branch points" are meaningless here. What does
# carry information is how curve-like the medial axis is: a tendrilled tumour
# thins to filaments, a compact one thins to a sheet.
SKEL_FEATURES = [
    "skel_size_norm",
    "skel_curve_frac",
    "skel_endpoints_norm",
    "skel_sheet_frac",
    "skel_complexity",
]
SPEC_FEATURES = [f"spharm_L{l}" for l in range(1, SPHARM_L + 1)] + ["spharm_rough"]

REGION_FEATURES = GRAN_FEATURES + CURV_FEATURES + SKEL_FEATURES + SPEC_FEATURES

LOC_FEATURES = [
    "svz_distance_norm",
    "svz_contact",
    "svz_contact_frac",
    "crosses_midline",
    "midline_minority_frac",
    "midline_distance_norm",
    "cortex_distance_norm",
    "depth_mean_norm",
    "depth_max_norm",
    "laterality_abs",
    "pos_ap",
    "pos_si",
    "pos_lat",
    "eccentricity_in_brain",
]
IC_FEATURES = [
    "rim_completeness",
    "rim_thickness_mean_norm",
    "rim_thickness_cv",
    "necrosis_eccentricity",
    "necrosis_buried_frac",
    "et_interface_frac",
    "et_surface_share",
    "edema_thickness_mean_norm",
    "edema_thickness_cv",
    "edema_thickness_max_norm",
    "edema_anisotropy",
    "edema_uncovered_frac",
    "core_offset_in_wt",
    "compartment_nesting",
]


# ---------------------------------------------------------------- helpers
def _surface(m):
    return m ^ ndi.binary_erosion(m)


def _principal_curvatures(m):
    """k1,k2 at surface voxels, from the shape operator of the level set."""
    sdf = ndi.distance_transform_edt(m).astype(np.float32) - ndi.distance_transform_edt(
        ~m
    ).astype(np.float32)
    sdf = ndi.gaussian_filter(sdf, 1.5)
    g = np.gradient(sdf)
    gm = np.sqrt(sum(x**2 for x in g)) + 1e-8
    H = np.empty(m.shape + (3, 3), np.float32)
    for i in range(3):
        gi = np.gradient(g[i])
        for j in range(3):
            H[..., i, j] = gi[j]
    s = _surface(m)
    idx = np.argwhere(s)
    if len(idx) < 20:
        return None, None
    if len(idx) > 40000:  # subsample very large surfaces
        idx = idx[np.random.default_rng(0).choice(len(idx), 40000, replace=False)]
    z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]
    n = np.stack([g[0][z, y, x], g[1][z, y, x], g[2][z, y, x]], -1) / gm[z, y, x, None]
    Hs = H[z, y, x] / gm[z, y, x, None, None]
    P = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
    S = -np.einsum("nij,njk,nkl->nil", P, Hs, P)
    ev = np.linalg.eigvalsh(S)
    # One eigenvalue lies along the surface normal and is ~0; the principal
    # curvatures are the two of largest magnitude. Taking min/max instead would
    # force k2 to 0 on any convex patch and collapse cap/cup into ridge/rut.
    order = np.argsort(-np.abs(ev), axis=1)[:, :2]
    k = np.take_along_axis(ev, order, axis=1)
    k = np.sort(k, axis=1)[:, ::-1]  # k1 >= k2
    return k[:, 0], k[:, 1]


def _radial_function(m, n_theta=48, n_phi=96):
    """Max radius of the mask along rays from its centroid, on a sphere grid."""
    pts = np.argwhere(m).astype(np.float32)
    c = pts.mean(0)
    rmax = float(np.linalg.norm(pts - c, axis=1).max()) + 2
    th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    ph = np.arange(n_phi) * 2 * np.pi / n_phi
    T, P = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.cos(T), np.sin(T) * np.cos(P), np.sin(T) * np.sin(P)], -1)
    t = np.linspace(0, rmax, 72, dtype=np.float32)
    q = c[None, None, None, :] + d[..., None, :] * t[None, None, :, None]
    qi = np.rint(q).astype(int)
    ok = np.all((qi >= 0) & (qi < np.array(m.shape)), axis=-1)
    hit = np.zeros(qi.shape[:-1], bool)
    hit[ok] = m[qi[ok][:, 0], qi[ok][:, 1], qi[ok][:, 2]]
    last = np.where(
        hit.any(-1),
        t[
            np.argmax(np.cumsum(hit[:, :, ::-1], -1)[:, :, ::-1] > 0, -1) * 0
            + (hit.shape[-1] - 1 - np.argmax(hit[:, :, ::-1], -1))
        ],
        0.0,
    )
    return last, th


def _spharm_power(r, th, L=SPHARM_L):
    """Rotation-invariant power per degree of the radial function."""
    n_theta, n_phi = r.shape
    ph = np.arange(n_phi) * 2 * np.pi / n_phi
    T, P = np.meshgrid(th, ph, indexing="ij")
    w = np.sin(T) * (np.pi / n_theta) * (2 * np.pi / n_phi)
    out = []
    c0 = None
    for l in range(0, L + 1):
        p = 0.0
        for mm in range(-l, l + 1):
            Y = sph_harm_y(l, mm, T, P)
            p += abs(np.sum(r * np.conj(Y) * w)) ** 2
        if l == 0:
            c0 = p if p > 0 else 1.0
        else:
            out.append(p / c0)  # normalised -> scale-invariant
    return out


# ---------------------------------------------------------------- family 3-6
def region_features(mask, min_voxels=200, spharm_on_raw=False):
    """spharm_on_raw=True computes the harmonic spectrum on the cropped original
    mask instead of the size-normalised copy. Audit (2026-08-24) showed the
    canonical resampling step destroys the rotation invariance the power spectrum
    should have: under an exact 90-degree rotation, deviation was 0.31/0.27/0.55
    at degrees 1/3/5 with resampling versus 0.000 without. The spectrum is already
    scale-invariant because every degree is normalised by the degree-0 power, so
    the resampling was never needed here. Default stays False so the v1 results
    remain exactly reproducible."""
    out = {k: np.nan for k in REGION_FEATURES}
    if mask.sum() < min_voxels:
        return out
    # Crop BEFORE canonicalising. _canonical caps its zoom by MAX_CANON_VOXELS /
    # m.size, i.e. by the size of the array it is handed. Passing the full
    # 155x240x240 volume made that cap bind for 53.7% of UPENN (case, region)
    # combinations -- 85.6% of necrotic cores -- so the mask never reached
    # CANON_R while every descriptor below still normalised by CANON_R. That
    # reintroduced the covert volume dependence _canonical exists to remove, and
    # it fell hardest on the smallest regions. shape_features.compute() has
    # always cropped first (see its line 141); this call site did not.
    # Protocol amendment 5, 2026-08-30.
    inner = _crop(mask)
    if inner is None:
        return out
    mc = _crop(_canonical(inner))
    if mc is None or mc.sum() < min_voxels:
        return out
    V = float(mc.sum())

    # --- granulometry: what survives opening at radii scaled to the object ---
    # Opening by a Euclidean ball of radius r, via two distance transforms
    # instead of an explicit structuring element -- exact and far cheaper.
    din = ndi.distance_transform_edt(mc)
    prev, half = 1.0, np.nan
    for r in GRAN_RADII:
        rad = r * CANON_R
        core = din >= rad
        opened = core if not core.any() else ndi.distance_transform_edt(~core) <= rad
        frac = float((opened & mc).sum() / V)
        out[f"gran_r{int(r * 100):02d}"] = frac
        if np.isnan(half) and frac < 0.5 <= prev:
            half = r
        prev = frac
    out["gran_half_radius"] = half if not np.isnan(half) else GRAN_RADII[-1]

    # --- curvature classes: pushing vs infiltrative margin -------------------
    k1, k2 = _principal_curvatures(mc)
    if k1 is not None:
        denom = k1 - k2
        S = np.where(
            np.abs(denom) > 1e-9,
            (2 / np.pi)
            * np.arctan((k2 + k1) / np.where(np.abs(denom) > 1e-9, denom, 1)),
            0.0,
        )
        edges = [-1.0, -0.625, -0.125, 0.125, 0.625, 1.0]
        for name, lo, hi in zip(CURV_CLASSES, edges[:-1], edges[1:]):
            out[f"curv_{name}_frac"] = float(np.mean((S >= lo) & (S < hi)))
        C = np.sqrt((k1**2 + k2**2) / 2) * CANON_R
        out["curvedness_mean"] = float(np.mean(C))
        out["curvedness_sd"] = float(np.std(C))

    # --- skeleton: tendrils and branching -----------------------------------
    sk = skeletonize(mc)
    nsk = int(sk.sum())
    if nsk > 5:
        nb = (
            ndi.convolve(
                sk.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant"
            )
            - sk
        )
        nbs = nb[sk]
        out["skel_size_norm"] = nsk / (CANON_R**2)  # sheet-like -> O(R^2)
        out["skel_curve_frac"] = float(np.mean(nbs <= 2))
        out["skel_sheet_frac"] = float(np.mean(nbs >= 6))
        out["skel_endpoints_norm"] = float((nbs == 1).sum()) / CANON_R
        pts = np.argwhere(sk).astype(float)
        span = np.linalg.norm(pts.max(0) - pts.min(0))
        out["skel_complexity"] = nsk / max(span**2, 1.0)

    # --- spherical-harmonic spectrum of the radial function ------------------
    try:
        r, th = _radial_function(_crop(mask) if spharm_on_raw else mc)
        pw = _spharm_power(r, th)
        for l, v in enumerate(pw, start=1):
            out[f"spharm_L{l}"] = float(v)
        out["spharm_rough"] = float(np.sum(pw[4:]))  # high-degree = fine detail
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- family 1
def location_features(
    seg, brain, vent, midline_x, labels_tc=(1, 3), labels_wt=(1, 2, 3)
):
    out = {k: np.nan for k in LOC_FEATURES}
    tc = np.isin(seg, labels_tc)
    wt = np.isin(seg, labels_wt)
    if tc.sum() < 50 or brain.sum() == 0:
        tc = wt
    if tc.sum() < 50:
        return out

    r_eq = (3 * float(tc.sum()) / (4 * np.pi)) ** (1 / 3)
    bidx = np.argwhere(brain)
    brain_r = float(np.linalg.norm(bidx - bidx.mean(0), axis=1).mean())

    # --- subventricular zone -------------------------------------------------
    dv = ndi.distance_transform_edt(~vent)
    dtc = dv[tc]
    out["svz_distance_norm"] = float(dtc.min() / brain_r)
    out["svz_contact"] = float(dtc.min() <= SVZ_CONTACT_MM)
    surf = _surface(tc)
    out["svz_contact_frac"] = float(np.mean(dv[surf] <= SVZ_CONTACT_MM))

    # --- midline -------------------------------------------------------------
    xs = np.argwhere(wt)[:, 2]
    left = float((xs < midline_x).sum())
    right = float((xs >= midline_x).sum())
    tot = left + right
    minority = min(left, right) / tot
    out["midline_minority_frac"] = minority
    out["crosses_midline"] = float(minority > 0.05)
    out["midline_distance_norm"] = float(np.abs(xs - midline_x).min() / brain_r)
    out["laterality_abs"] = float(abs(left - right) / tot)

    # --- cortex / depth ------------------------------------------------------
    db = ndi.distance_transform_edt(brain)
    out["cortex_distance_norm"] = float(db[tc].max() / brain_r)  # deepest point
    out["depth_mean_norm"] = float(db[tc].mean() / brain_r)
    out["depth_max_norm"] = float(db[tc].max() / brain_r)

    # --- position within the brain frame ------------------------------------
    lo, hi = bidx.min(0), bidx.max(0)
    c = np.argwhere(tc).mean(0)
    rel = (c - lo) / np.maximum(hi - lo, 1)
    out["pos_si"], out["pos_ap"] = float(rel[0]), float(rel[1])
    out["pos_lat"] = float(abs(c[2] - midline_x) / brain_r)
    out["eccentricity_in_brain"] = float(np.linalg.norm(c - bidx.mean(0)) / brain_r)
    return out


# ---------------------------------------------------------------- family 2
def intercompartment_features(seg, l_netc=1, l_snfh=2, l_et=3):
    out = {k: np.nan for k in IC_FEATURES}
    netc, snfh, et = seg == l_netc, seg == l_snfh, seg == l_et
    tc, wt = netc | et, netc | snfh | et
    if tc.sum() < 100:
        return out
    r_tc = (3 * float(tc.sum()) / (4 * np.pi)) ** (1 / 3)

    # --- enhancing rim around the necrotic core ------------------------------
    if netc.sum() >= 50 and et.sum() >= 50:
        ns = _surface(netc)
        d_et = ndi.distance_transform_edt(~et)
        out["rim_completeness"] = float(np.mean(d_et[ns] <= 3.0))
        th = ndi.distance_transform_edt(et)[et]
        out["rim_thickness_mean_norm"] = float(th.mean() / r_tc)
        out["rim_thickness_cv"] = float(th.std() / max(th.mean(), 1e-6))
        cn = np.argwhere(netc).mean(0)
        ce = np.argwhere(et).mean(0)
        out["necrosis_eccentricity"] = float(np.linalg.norm(cn - ce) / r_tc)
        d_out = ndi.distance_transform_edt(~(snfh | et))
        out["necrosis_buried_frac"] = float(np.mean(d_out[ns] > 1.5))
        inter = ns & (d_et <= 1.5)
        out["et_interface_frac"] = float(inter.sum() / max(_surface(et).sum(), 1))

    if et.sum() >= 50:
        out["et_surface_share"] = float(
            np.sum(_surface(wt) & et) / max(_surface(wt).sum(), 1)
        )

    # --- oedema shell outside the core ---------------------------------------
    if snfh.sum() >= 100:
        d_tc = ndi.distance_transform_edt(~tc)
        ws = _surface(wt)
        ext = d_tc[ws]
        out["edema_thickness_mean_norm"] = float(ext.mean() / r_tc)
        out["edema_thickness_cv"] = float(ext.std() / max(ext.mean(), 1e-6))
        out["edema_thickness_max_norm"] = float(ext.max() / r_tc)
        out["edema_uncovered_frac"] = float(np.mean(ext <= 1.5))
        # directional anisotropy: spread of oedema extent across solid-angle bins
        c = np.argwhere(tc).mean(0)
        v = np.argwhere(ws).astype(float) - c
        n = np.linalg.norm(v, axis=1) + 1e-6
        u = v / n[:, None]
        oct_id = (
            (u[:, 0] > 0).astype(int) * 4
            + (u[:, 1] > 0).astype(int) * 2
            + (u[:, 2] > 0).astype(int)
        )
        means = [ext[oct_id == k].mean() for k in range(8) if (oct_id == k).sum() > 5]
        if len(means) >= 4:
            out["edema_anisotropy"] = float(np.std(means) / max(np.mean(means), 1e-6))
        out["core_offset_in_wt"] = float(
            np.linalg.norm(c - np.argwhere(wt).mean(0)) / r_tc
        )

    # --- nesting: how completely each compartment sits inside the next -------
    if netc.sum() >= 50:
        hull_ok = float(np.sum(netc & ndi.binary_dilation(et, iterations=3))) / max(
            netc.sum(), 1
        )
        out["compartment_nesting"] = hull_ok
    return out
