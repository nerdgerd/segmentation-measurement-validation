"""Patient sufficient moments make fixed-battery repetition audits tractable.

Callers supply audited structure and explicit Boolean availability, never old CCCs.
Replicate ranges have absolute seed keys, so independent batches can be resumed and
combined without changing the coefficient battery or replacing failed repetitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest

from .provenance import content_hash, rng
from .simulation import (
    Template,
    coefficient_battery,
    make_laws,
    population_moments,
    hierarchical_draws,
    coverage_fixture,
)


@dataclass(frozen=True)
class SimulationSupport:
    reference_available: np.ndarray
    real_available: np.ndarray
    synthetic_available: np.ndarray | None = None

    def validate(self, template, *, require_synthetic=False, draw_count=100):
        template.validate()
        u, j = len(template.patients), len(template.descriptor_ids)
        specifications = [
            (self.reference_available, (u, j)),
            (self.real_available, (sum(template.alternate_counts), j)),
        ]
        if require_synthetic:
            specifications.append((self.synthetic_available, (u, draw_count, j)))
        for array, shape in specifications:
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != bool
                or array.shape != shape
            ):
                raise ValueError(
                    f"Explicit Boolean availability with shape {shape} is required"
                )

    @property
    def identity(self):
        digests = []
        for array in [
            self.reference_available,
            self.real_available,
            self.synthetic_available,
        ]:
            digests.append(
                None
                if array is None
                else {
                    "shape": array.shape,
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(array).tobytes()
                    ).hexdigest(),
                }
            )
        return content_hash(digests)


class PatientMoments:
    """Precompute conditional reader/nodule weights, retaining nominal coverage."""

    def __init__(
        self,
        template,
        row_units,
        availability,
        reference_available,
        *,
        min_patients=10,
        min_coverage=0.90,
    ):
        self.row_units = np.asarray(row_units, dtype=int)
        patient_ids = sorted(set(template.patients))
        self.unit_patient = np.asarray(
            [patient_ids.index(p) for p in template.patients]
        )
        self.row_patient = self.unit_patient[self.row_units]
        self.row_order = np.argsort(self.row_units, kind="stable")
        ordered_units = self.row_units[self.row_order]
        self.unit_starts = np.r_[0, np.flatnonzero(np.diff(ordered_units)) + 1]
        self.present_units = ordered_units[self.unit_starts]
        self.rows_already_ordered = bool(np.all(np.diff(self.row_units) >= 0))
        self.patient_count = len(patient_ids)
        self.feature_count = len(template.descriptor_ids)
        self.min_coverage = min_coverage
        u, j, p = len(template.patients), self.feature_count, self.patient_count
        available = availability & reference_available[self.row_units]
        unit_count = np.zeros((u, j))
        np.add.at(unit_count, self.row_units, available)
        patient_unit_count = np.zeros((p, j))
        np.add.at(patient_unit_count, self.unit_patient, unit_count > 0)
        self.present = patient_unit_count > 0
        denominator = unit_count[self.row_units] * patient_unit_count[self.row_patient]
        self.weights = np.divide(
            available,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        self.reference_unit_weights = np.zeros((u, j))
        np.add.at(self.reference_unit_weights, self.row_units, self.weights)
        self.reference_unit_weights = np.divide(
            self.reference_unit_weights,
            self.present.sum(axis=0),
            out=np.zeros_like(self.reference_unit_weights),
            where=self.present.sum(axis=0) > 0,
        )
        original_unit_count = np.bincount(self.unit_patient, minlength=p)
        original_measurement_count = np.bincount(self.row_units, minlength=u)
        original_weight = 1.0 / (
            original_unit_count[self.row_patient]
            * original_measurement_count[self.row_units]
        )
        self.coverage = np.zeros((p, j))
        np.add.at(self.coverage, self.row_patient, original_weight[:, None] * available)
        self.original_eligible = (self.present.sum(axis=0) >= min_patients) & (
            self.coverage.mean(axis=0) >= min_coverage
        )

    def integrate(self, references, alternates):
        x = references[self.row_units]
        delta = alternates - x
        output = np.zeros((self.patient_count, self.feature_count, 5))
        for column, values in enumerate([x, delta, x * x, delta * delta, x * delta]):
            weighted = values * self.weights
            ordered = (
                weighted if self.rows_already_ordered else weighted[self.row_order]
            )
            by_unit = np.add.reduceat(ordered, self.unit_starts, axis=0)
            np.add.at(
                output[:, :, column], self.unit_patient[self.present_units], by_unit
            )
        return output

    def pool(self, patient_moments, counts=None):
        if counts is None:
            counts = np.ones((1, self.patient_count))
        counts = np.atleast_2d(np.asarray(counts, dtype=float))
        denominator = counts @ self.present.astype(float)
        totals = counts @ patient_moments.reshape(self.patient_count, -1)
        totals = totals.reshape(len(counts), self.feature_count, 5)
        values = np.divide(
            totals,
            denominator[:, :, None],
            out=np.full_like(totals, np.nan),
            where=denominator[:, :, None] > 0,
        )
        mu, b, xx, dd, xd = np.moveaxis(values, -1, 0)
        v, q, a = xx - mu * mu, dd - b * b, xd - mu * b
        # These are simulations with well-scaled Gaussian values; only relative roundoff is removed.
        for variance, second in [(v, xx), (q, dd)]:
            tiny = (variance < 0) & (variance >= -1e-12 * np.abs(second))
            variance[tiny] = 0.0
        mse = q + b * b
        ccc_denominator = 2 * (v + a) + mse
        coverage = (counts @ self.coverage) / counts.sum(axis=1)[:, None]
        valid = (
            self.original_eligible[None, :]
            & (coverage >= self.min_coverage)
            & (v > 0)
            & (q >= 0)
            & (ccc_denominator > 0)
        )
        ccc = np.divide(
            2 * (v + a), ccc_denominator, out=np.full_like(v, np.nan), where=valid
        )
        return {
            "reference_mean": mu,
            "reference_variance": v,
            "disagreement_mean": b,
            "disagreement_variance": q,
            "reference_disagreement_covariance": a,
            "disagreement_mse": mse,
            "ccc": ccc,
            "coverage": coverage,
        }


def _correlation(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _mean_absolute(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(np.abs(finite))) if len(finite) else np.nan


def _weighted_iqr(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0:
        return np.nan
    cumulative /= cumulative[-1]
    positions = np.searchsorted(cumulative, [0.25, 0.75], side="left")
    return float(values[order[positions[1]]] - values[order[positions[0]]])


def finite_sample_tables(
    cfg,
    template: Template,
    support: SimulationSupport,
    *,
    regime,
    theta,
    start=0,
    stop=None,
    stream="mechanism_final",
):
    stop = (
        cfg["simulation"]["final_mechanism_replicates_per_cell"]
        if stop is None
        else stop
    )
    draw_count = cfg["synthetic"]["draw_count_initial"]
    support.validate(template, require_synthetic=True, draw_count=draw_count)
    if not 0 <= start < stop:
        raise ValueError("A nonempty absolute replicate range is required")
    if (
        regime not in cfg["simulation"]["core_regimes"]
        or theta not in cfg["simulation"]["finite_theta_grid"]
    ):
        raise ValueError("Cell is not part of the declared finite mechanism grid")
    battery = coefficient_battery(
        cfg,
        template.descriptor_ids,
        template.families,
        template_key=template.name,
        stream=stream,
    )
    real, syn = make_laws(battery, h=1.0, theta=theta, regime=regime)
    truth_r, truth_s = population_moments(real), population_moments(syn)
    u, j = len(template.patients), len(template.descriptor_ids)
    real_units = np.repeat(np.arange(u), template.alternate_counts)
    syn_units = np.repeat(np.arange(u), draw_count)
    options = {
        "min_patients": cfg["features"]["min_evaluation_independent_patients"],
        "min_coverage": cfg["features"]["min_evaluation_weighted_support"],
    }
    rp = PatientMoments(
        template,
        real_units,
        support.real_available,
        support.reference_available,
        **options,
    )
    sp = PatientMoments(
        template,
        syn_units,
        support.synthetic_available.reshape(-1, j),
        support.reference_available,
        **options,
    )
    real_unit_defined = np.zeros((u, j), dtype=bool)
    np.logical_or.at(real_unit_defined, real_units, support.real_available)
    diagnostic_reference_available = (
        support.reference_available
        & real_unit_defined
        & support.synthetic_available.any(axis=1)
    )
    if np.array_equal(diagnostic_reference_available, support.reference_available):
        diagnostic_real, diagnostic_synthetic = rp, sp
    else:
        diagnostic_real = PatientMoments(
            template,
            real_units,
            support.real_available,
            diagnostic_reference_available,
            **options,
        )
        diagnostic_synthetic = PatientMoments(
            template,
            syn_units,
            support.synthetic_available.reshape(-1, j),
            diagnostic_reference_available,
            **options,
        )
    template_hash, support_hash = template.identity, support.identity
    rows = []
    for replicate in range(start, stop):
        data = hierarchical_draws(
            cfg,
            template,
            real,
            syn,
            stream=stream,
            replicate=replicate,
            draw_count=draw_count,
        )
        r = {
            key: value[0]
            for key, value in rp.pool(
                rp.integrate(data["reference"], data["real"])
            ).items()
        }
        s = {
            key: value[0]
            for key, value in sp.pool(
                sp.integrate(data["reference"], data["synthetic"].reshape(-1, j))
            ).items()
        }
        valid = np.isfinite(r["ccc"]) & np.isfinite(s["ccc"])
        rank_r, rank_s = rankdata(r["ccc"][valid]), rankdata(s["ccc"][valid])
        dr = (
            r
            if diagnostic_real is rp
            else {
                key: value[0]
                for key, value in diagnostic_real.pool(
                    diagnostic_real.integrate(data["reference"], data["real"])
                ).items()
            }
        )
        ds = (
            s
            if diagnostic_synthetic is sp
            else {
                key: value[0]
                for key, value in diagnostic_synthetic.pool(
                    diagnostic_synthetic.integrate(
                        data["reference"], data["synthetic"].reshape(-1, j)
                    )
                ).items()
            }
        )
        diagnostic_valid = np.isfinite(dr["ccc"]) & np.isfinite(ds["ccc"])
        ratio = 0.5 * np.log2(
            ds["disagreement_mse"][diagnostic_valid]
            / dr["disagreement_mse"][diagnostic_valid]
        )
        iqr = np.asarray(
            [
                _weighted_iqr(data["reference"][:, d], rp.reference_unit_weights[:, d])
                for d in np.flatnonzero(valid)
            ]
        )
        partial = np.nan
        if valid.sum() >= 3 and np.ptp(iqr) > 0:
            design = np.column_stack([np.ones(len(iqr)), rankdata(iqr)])
            residual_r = (
                rank_r - design @ np.linalg.lstsq(design, rank_r, rcond=None)[0]
            )
            residual_s = (
                rank_s - design @ np.linalg.lstsq(design, rank_s, rcond=None)[0]
            )
            rank_scale = max(np.linalg.norm(rank_r), np.linalg.norm(rank_s))
            if (
                min(np.linalg.norm(residual_r), np.linalg.norm(residual_s))
                > 1e-12 * rank_scale
            ):
                partial = _correlation(residual_r, residual_s)
        rows.append(
            {
                "template": template.name,
                "template_sha256": template_hash,
                "availability_sha256": support_hash,
                "regime": regime,
                "theta": theta,
                "h": 1.0,
                "replicate": replicate,
                "final_simulation_estimate": stream != "smoke",
                "descriptor_count": j,
                "evaluable_descriptor_count": int(valid.sum()),
                "matched_disagreement_descriptor_count": int(diagnostic_valid.sum()),
                "patient_count": rp.patient_count,
                "sample_profile_spearman": _correlation(rank_r, rank_s),
                "population_profile_spearman": _correlation(
                    rankdata(truth_r["ccc"]), rankdata(truth_s["ccc"])
                ),
                "legacy_partial_iqr_diagnostic": partial,
                "median_signed_log2_rmse_ratio": float(np.median(ratio))
                if len(ratio)
                else np.nan,
                "median_absolute_log2_rmse_ratio": float(np.median(np.abs(ratio)))
                if len(ratio)
                else np.nan,
                "real_mean_absolute_ccc_estimation_error": _mean_absolute(
                    r["ccc"] - truth_r["ccc"]
                ),
                "synthetic_mean_absolute_ccc_estimation_error": _mean_absolute(
                    s["ccc"] - truth_s["ccc"]
                ),
                "real_undefined_fraction": float(np.mean(~np.isfinite(r["ccc"]))),
                "synthetic_undefined_fraction": float(np.mean(~np.isfinite(s["ccc"]))),
            }
        )
    truth = pd.DataFrame(
        {
            "descriptor_id": template.descriptor_ids,
            "family": template.families,
            "real_population_ccc": truth_r["ccc"],
            "synthetic_population_ccc": truth_s["ccc"],
        }
    )
    truth["template"], truth["regime"], truth["theta"] = template.name, regime, theta
    return {
        "finite_sample_mechanism": pd.DataFrame(rows),
        "finite_mechanism_truth": truth,
    }


def coverage_tables(
    cfg,
    template: Template,
    support: SimulationSupport,
    *,
    regime,
    start=0,
    stop=None,
    stream="interval_audit",
):
    stop = (
        cfg["simulation"]["interval_audit_datasets_per_cell"] if stop is None else stop
    )
    support.validate(template)
    if (
        not 0 <= start < stop
        or regime not in cfg["simulation"]["interval_audit_regimes"]
    ):
        raise ValueError("Invalid coverage cell or absolute replicate range")
    battery = coefficient_battery(
        cfg,
        template.descriptor_ids,
        template.families,
        template_key=template.name,
        stream=stream,
    )
    law, _ = make_laws(battery, h=1, theta=0, regime=regime)
    fixture, selected_sets, utilities = coverage_fixture(law, template.descriptor_ids)
    truth = population_moments(fixture)["ccc"]
    names = [f"ccc:{d}" for d in template.descriptor_ids]
    truth_values = list(truth)
    indices = {
        name: [template.descriptor_ids.index(d) for d in selection]
        for name, selection in selected_sets.items()
    }
    for name in selected_sets:
        names.append("utility:" + name)
        truth_values.append(utilities[name])
    for name in ["second", "higher", "lower"]:
        names.append(f"contrast:{name}_minus_first")
        truth_values.append(utilities[name] - utilities["first"])
    truth_values = np.asarray(truth_values)
    units = np.repeat(np.arange(len(template.patients)), template.alternate_counts)
    accumulator = PatientMoments(
        template,
        units,
        support.real_available,
        support.reference_available,
        min_patients=cfg["features"]["min_evaluation_independent_patients"],
        min_coverage=cfg["features"]["min_evaluation_weighted_support"],
    )
    n, boot = (
        accumulator.patient_count,
        cfg["simulation"]["bootstrap_replicates_per_interval_audit_dataset"],
    )
    maximum_undefined = cfg["uncertainty"][
        "max_undefined_bootstrap_fraction_for_nominal_interval"
    ]
    alpha = 1 - cfg["uncertainty"]["nominal_pointwise_level"]
    template_hash, support_hash = template.identity, support.identity

    def endpoints(ccc):
        outputs = [ccc]
        utility = {
            name: np.mean(ccc[:, index], axis=1) for name, index in indices.items()
        }
        outputs.append(np.column_stack(list(utility.values())))
        outputs.append(
            np.column_stack(
                [
                    utility[name] - utility["first"]
                    for name in ["second", "higher", "lower"]
                ]
            )
        )
        return np.column_stack(outputs)

    rows = []
    for replicate in range(start, stop):
        data = hierarchical_draws(
            cfg,
            template,
            fixture,
            fixture,
            stream=stream,
            replicate=replicate,
            draw_count=1,
        )
        sufficient = accumulator.integrate(data["reference"], data["real"])
        point = endpoints(accumulator.pool(sufficient)["ccc"])[0]
        counts = rng(
            cfg, stream, "coverage_bootstrap", template.identity, regime, replicate
        ).multinomial(n, np.full(n, 1 / n), size=boot)
        boot_values = endpoints(accumulator.pool(sufficient, counts)["ccc"])
        for e, name in enumerate(names):
            available = boot_values[:, e][np.isfinite(boot_values[:, e])]
            undefined = 1 - len(available) / boot
            lower, upper = (
                np.quantile(available, [alpha / 2, 1 - alpha / 2])
                if len(available)
                else (np.nan, np.nan)
            )
            issued = bool(np.isfinite(point[e]) and undefined <= maximum_undefined)
            covered = bool(lower <= truth_values[e] <= upper) if issued else False
            rows.append(
                {
                    "template": template.name,
                    "template_sha256": template_hash,
                    "availability_sha256": support_hash,
                    "regime": regime,
                    "replicate": replicate,
                    "endpoint": name,
                    "patient_count": n,
                    "population_truth": truth_values[e],
                    "point_estimate": point[e],
                    "percentile_lower": lower,
                    "percentile_upper": upper,
                    "interval_issued": issued,
                    "covered": covered,
                    "undefined_bootstrap_fraction": undefined,
                    "bootstrap_replicates": boot,
                    "final_simulation_estimate": stream != "smoke",
                }
            )
    fixed_sets = pd.DataFrame(
        [
            {"set": name, "descriptor_id": d, "population_mean_ccc": utilities[name]}
            for name, selected in selected_sets.items()
            for d in selected
        ]
    )
    fixed_sets["template"], fixed_sets["regime"] = template.name, regime
    return {
        "coverage_repetitions": pd.DataFrame(rows),
        "coverage_fixed_sets": fixed_sets,
    }


def summarize_coverage(repetitions, *, nominal=0.95, material_shortfall=0.02):
    """Unissued intervals count as failures; conditional coverage is also explicit."""
    if repetitions.duplicated(["template", "regime", "replicate", "endpoint"]).any():
        raise ValueError(
            "Coverage batches overlap; duplicate repetitions cannot be pooled"
        )
    rows = []
    for (template, regime, endpoint), group in repetitions.groupby(
        ["template", "regime", "endpoint"], sort=True
    ):
        successes, total = int(group.covered.sum()), len(group)
        interval = binomtest(successes, total).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        issued = group[group.interval_issued]
        coverage = successes / total
        rows.append(
            {
                "template": template,
                "regime": regime,
                "endpoint": endpoint,
                "datasets": total,
                "coverage_all_datasets": coverage,
                "coverage_among_issued_intervals": float(issued.covered.mean())
                if len(issued)
                else np.nan,
                "coverage_monte_carlo_lower": interval.low,
                "coverage_monte_carlo_upper": interval.high,
                "mean_bias": float(
                    (group.point_estimate - group.population_truth).mean()
                ),
                "mean_interval_width_when_issued": float(
                    (issued.percentile_upper - issued.percentile_lower).mean()
                ),
                "interval_failure_fraction": 1 - len(issued) / total,
                "mean_undefined_bootstrap_fraction": float(
                    group.undefined_bootstrap_fraction.mean()
                ),
                "material_coverage_shortfall": coverage < nominal - material_shortfall,
            }
        )
    return pd.DataFrame(rows)
