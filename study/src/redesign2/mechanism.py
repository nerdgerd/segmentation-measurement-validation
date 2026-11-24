"""Descriptive validation diagnostics with definition and information boundaries.

These functions preserve whole disagreement laws and region clusters. Matching
failures, incomplete batteries and finite-sample failures are outputs, not reasons
to change an endpoint. No function has filesystem access or hidden cohort inputs.
"""

from dataclasses import replace
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import rankdata, norm
from .simulation import make_laws, population_moments


def iqr(x):
    return float(np.diff(np.quantile(np.asarray(x), [0.25, 0.75]))[0])


def association(x, y):
    """Spearman correlation across features on their common finite support."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    a, b = rankdata(np.asarray(x)[valid]), rankdata(np.asarray(y)[valid])
    return float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan


def distribution(x):
    a = np.asarray(x, dtype=float)
    v = a[np.isfinite(a)]
    return dict(
        datasets=len(a),
        defined=len(v),
        undefined_fraction=1 - len(v) / len(a) if len(a) else np.nan,
        median=float(np.median(v)) if len(v) else np.nan,
        q25=float(np.quantile(v, 0.25)) if len(v) else np.nan,
        q75=float(np.quantile(v, 0.75)) if len(v) else np.nan,
        p025=float(np.quantile(v, 0.025)) if len(v) else np.nan,
        p975=float(np.quantile(v, 0.975)) if len(v) else np.nan,
    )


def match_law(
    battery,
    reference_mean,
    reference_variance,
    disagreement_mse,
    *,
    law,
    theta,
    side,
    grid=(0, 12, 601),
    tolerance=1e-6,
):
    """Match model medians and IQRs to one observed or generated moment list.

    The matching variables are log(v / mu²) and log(M / mu²). The whole
    offset/slope/random-scale law is multiplied together, preserving its
    covariance contribution to CCC. The first root on the fixed grid is used;
    a missing root is reported without changing the grid or model.
    """
    mu, v, D = map(
        lambda x: np.asarray(x, float),
        (reference_mean, reference_variance, disagreement_mse),
    )
    refvalid = np.isfinite(mu) & (mu != 0) & np.isfinite(v) & (v > 0)
    dvalid = refvalid & np.isfinite(D) & (D > 0)
    if (
        refvalid.sum() < 4
        or dvalid.sum() < 4
        or iqr(battery.reference.to_numpy()[refvalid]) == 0
    ):
        return None, dict(
            matched=False, reason="insufficient_defined_matching_marginal"
        )
    rv = np.log(v[refvalid]) - 2 * np.log(abs(mu[refvalid]))
    dd = np.log(D[dvalid]) - 2 * np.log(abs(mu[dvalid]))
    g = battery.reference.to_numpy()
    h = iqr(rv) / (2 * iqr(g[refvalid]))
    real, synthetic = make_laws(battery, h=h, theta=theta, regime=law)
    base = real if side == "observed" else synthetic
    matched_v = np.exp(np.median(rv) + 2 * h * (g - np.median(g[refvalid])))
    base = replace(base, source_mean=np.ones(len(g)), source_variance=matched_v)
    u = (
        battery.real_scale.to_numpy()
        if side == "observed"
        else theta * battery.real_scale.to_numpy()
        + np.sqrt(1 - theta**2) * battery.independent_scale.to_numpy()
    )
    target = iqr(dd)

    def scaled(lam, common=1.0):
        multiplier = common * np.exp(0.5 * (lam - 1) * u)
        return replace(
            base,
            bias=base.bias * multiplier,
            slope=base.slope * multiplier,
            residual_sd=base.residual_sd * multiplier,
        )

    def error(lam):
        return (
            iqr(np.log(population_moments(scaled(lam))["disagreement_mse"][dvalid]))
            - target
        )

    points = np.linspace(*grid)
    errors = np.array([error(lam) for lam in points])
    roots = np.flatnonzero(abs(errors) <= tolerance)
    lam = float(points[roots[0]]) if len(roots) else None
    if lam is None:
        brackets = np.flatnonzero(errors[:-1] * errors[1:] < 0)
        if len(brackets):
            j = brackets[0]
            lam = float(brentq(error, points[j], points[j + 1], xtol=1e-11))
    rec = dict(
        h=h,
        reference_log_median=float(np.median(rv)),
        reference_log_iqr=iqr(rv),
        disagreement_log_median=float(np.median(dd)),
        disagreement_log_iqr=target,
        reference_matching_count=int(refvalid.sum()),
        disagreement_matching_count=int(dvalid.sum()),
        calibration_grid_min_error=float(errors.min()),
        calibration_grid_max_error=float(errors.max()),
    )
    if lam is None:
        return None, dict(
            rec, matched=False, reason="no_root_in_frozen_disagreement_multiplier_grid"
        )
    before = population_moments(scaled(lam))["disagreement_mse"]
    common = float(np.exp(0.5 * (np.median(dd) - np.median(np.log(before[dvalid])))))
    result = scaled(lam, common)
    actual = population_moments(result)
    rec.update(
        lambda_multiplier=lam,
        common_multiplier=common,
        matched=True,
        reason="",
        attained_reference_log_median=float(np.median(np.log(matched_v[refvalid]))),
        attained_reference_log_iqr=iqr(np.log(matched_v[refvalid])),
        attained_disagreement_log_median=float(
            np.median(np.log(actual["disagreement_mse"][dvalid]))
        ),
        attained_disagreement_log_iqr=iqr(np.log(actual["disagreement_mse"][dvalid])),
    )
    assert abs(rec["attained_disagreement_log_iqr"] - target) <= 2 * tolerance
    return result, rec


def permutation_blocks(ids, families, regions):
    """Keep the same feature's region vector together within its family."""
    clusters = {}
    for j, (d, f, r) in enumerate(zip(ids, families, regions, strict=True)):
        stem = (
            d.split("__", 1)[1]
            if "__" in d and d.split("__", 1)[0] in {"WT", "TC", "ET", "NETC"}
            else d
        )
        clusters.setdefault((f, stem), {})[r] = j
    blocks = {}
    for (f, stem), vector in clusters.items():
        blocks.setdefault((f, tuple(sorted(vector))), []).append((stem, vector))
    return blocks


def permutation_indices(
    ids, families, regions, *, repetitions, rng, unrestricted=False
):
    result = np.tile(np.arange(len(ids)), (repetitions, 1))
    records = []
    if unrestricted:
        return np.argsort(rng.random(result.shape), axis=1), [
            dict(
                family="all",
                region_signature="unrestricted",
                cluster_count=len(ids),
                distinct_permutations=str(math.factorial(len(ids))),
            )
        ]
    for (family, signature), clusters in sorted(
        permutation_blocks(ids, families, regions).items()
    ):
        order = np.argsort(rng.random((repetitions, len(clusters))), axis=1)
        for r in signature:
            positions = np.array([v[r] for _, v in clusters])
            result[:, positions] = positions[order]
        records.append(
            dict(
                family=family,
                region_signature="|".join(signature),
                cluster_count=len(clusters),
                descriptors=sum(len(v) for _, v in clusters),
                members="|".join(s for s, _ in clusters),
                distinct_permutations=str(math.factorial(len(clusters))),
                fixed=len(clusters) == 1,
            )
        )
    return result, records


def permuted_associations(real_ccc, syn_variance, syn_a, syn_D, indices):
    av = np.asarray(syn_a) / syn_variance
    dv = np.asarray(syn_D) / syn_variance
    numerator = 2 * (1 + av[indices])
    denominator = numerator + dv[indices]
    ccc = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    rr = rankdata(real_ccc)
    rr = rr - rr.mean()
    sr = rankdata(ccc, axis=1)
    sr = sr - sr.mean(axis=1)[:, None]
    out = (sr @ rr) / np.sqrt(np.sum(sr * sr, axis=1) * np.sum(rr * rr))
    out[~np.isfinite(ccc).all(axis=1)] = np.nan
    return out


def coverage_prediction(r):
    r = np.asarray(r, dtype=float)
    return norm.cdf(1.959964 - 2 * r) - norm.cdf(-1.959964 - 2 * r)


def top_k(ids, scores):
    if not np.isfinite(scores).all():
        return None
    k = max(1, len(ids) // 4)
    return tuple(sorted(range(len(ids)), key=lambda j: (-float(scores[j]), ids[j]))[:k])
