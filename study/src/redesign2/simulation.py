"""Known Gaussian laws separate case-mix interventions from measurement units.

Coefficient fields stay fixed as reference distributions and patient samples change.
The constrained coverage fixture is labeled separately: exact equal utility is a
constructed truth, and is not obtained by searching the unconstrained mechanism law.
No function reads cohort values, discovers a template, or writes files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .moments import JointMoments, ReferenceMoments
from .provenance import content_hash, rng
from .transport import fit_source_model, predict_target


@dataclass(frozen=True)
class PopulationLaw:
    source_mean: np.ndarray
    source_variance: np.ndarray
    bias: np.ndarray
    slope: np.ndarray
    residual_sd: np.ndarray
    heteroscedastic_log_sd_slope: float = 0.0


@dataclass(frozen=True)
class Template:
    """One metadata row per nodule, never one repeated row per observed reader."""

    name: str
    descriptor_ids: tuple[str, ...]
    families: tuple[str, ...]
    patients: tuple[str, ...]
    nodules: tuple[str, ...]
    alternate_counts: tuple[int, ...]

    def validate(self):
        if not self.descriptor_ids or len(set(self.descriptor_ids)) != len(
            self.descriptor_ids
        ):
            raise ValueError("Distinct stable descriptor identifiers are required")
        if len(self.families) != len(self.descriptor_ids) or any(
            not f for f in self.families
        ):
            raise ValueError("An audited family is required for every descriptor")
        n = len(self.patients)
        if not n or len(self.nodules) != n or len(self.alternate_counts) != n:
            raise ValueError("Patient/nodule/count metadata have inconsistent lengths")
        if len(set(zip(self.patients, self.nodules))) != n:
            raise ValueError("A patient/nodule unit occurs more than once")
        if any(int(k) != k or k < 1 for k in self.alternate_counts):
            raise ValueError("Every nodule requires a positive integer alternate count")

    @property
    def identity(self):
        self.validate()
        return content_hash(self.__dict__)


def _family_field(cfg, stream, template_key, kind, descriptor_ids, families):
    family_values = {
        family: rng(
            cfg, stream, "coefficient", template_key, kind, "family", family
        ).normal()
        for family in sorted(set(families))
    }
    return np.asarray(
        [
            (
                family_values[family]
                + rng(
                    cfg,
                    stream,
                    "coefficient",
                    template_key,
                    kind,
                    "descriptor",
                    descriptor,
                ).normal()
            )
            / np.sqrt(2)
            for descriptor, family in zip(descriptor_ids, families, strict=True)
        ]
    )


def coefficient_battery(
    cfg, descriptor_ids, families, *, template_key, stream="mechanism_final"
) -> pd.DataFrame:
    descriptor_ids, families = tuple(descriptor_ids), tuple(families)
    if len(descriptor_ids) != len(families) or len(set(descriptor_ids)) != len(
        descriptor_ids
    ):
        raise ValueError(
            "Descriptor identities and families must be explicit and unique"
        )
    if not descriptor_ids or any(not f for f in families):
        raise ValueError("Empty descriptor universe or unknown family")
    fields = {
        kind: _family_field(cfg, stream, template_key, kind, descriptor_ids, families)
        for kind in [
            "reference",
            "real_scale",
            "independent_scale",
            "real_bias",
            "synthetic_bias",
            "real_slope",
            "synthetic_slope",
            "changed_bias",
            "changed_slope",
            "changed_scale",
        ]
    }
    return pd.DataFrame({"descriptor_id": descriptor_ids, "family": families, **fields})


def make_laws(
    battery: pd.DataFrame,
    *,
    h: float,
    theta: float,
    regime: str,
    heteroscedastic_slope=0.3,
) -> tuple[PopulationLaw, PopulationLaw]:
    if not -1 <= theta <= 1 or h < 0:
        raise ValueError("Invalid coupling or reference dispersion")
    if regime not in {"additive", "biased_proportional", "heteroscedastic"}:
        raise ValueError(f"Unknown disagreement regime: {regime}")
    n = len(battery)
    variance = np.exp(2 * h * battery.reference.to_numpy())
    u_real = battery.real_scale.to_numpy()
    u_syn = (
        theta * u_real + np.sqrt(1 - theta**2) * battery.independent_scale.to_numpy()
    )
    laws = []
    for name, log_scale in [("real", u_real), ("synthetic", u_syn)]:
        bias = (
            np.zeros(n)
            if regime == "additive"
            else 0.25 * battery[f"{name}_bias"].to_numpy()
        )
        slope = (
            np.zeros(n)
            if regime == "additive"
            else 0.25 * np.tanh(battery[f"{name}_slope"].to_numpy())
        )
        laws.append(
            PopulationLaw(
                np.zeros(n),
                variance.copy(),
                bias,
                slope,
                0.5 * np.exp(0.5 * log_scale),
                heteroscedastic_slope if regime == "heteroscedastic" else 0.0,
            )
        )
    return tuple(laws)


def population_moments(law: PopulationLaw, target_mean=None, target_variance=None):
    """Integrate the source-centered conditional law over a Gaussian target X."""
    mu = (
        law.source_mean if target_mean is None else np.asarray(target_mean, dtype=float)
    )
    v = (
        law.source_variance
        if target_variance is None
        else np.asarray(target_variance, dtype=float)
    )
    mu, v = np.broadcast_arrays(mu, v)
    if np.any(law.source_variance <= 0) or np.any(v < 0):
        raise ValueError(
            "Source variance must be positive and target variance nonnegative"
        )
    shift = mu - law.source_mean
    kappa = law.heteroscedastic_log_sd_slope
    residual_second_moment = law.residual_sd**2 * np.exp(
        2 * kappa * shift / np.sqrt(law.source_variance)
        + 2 * kappa**2 * v / law.source_variance
    )
    b = law.bias + law.slope * shift
    a = law.slope * v
    q = law.slope**2 * v + residual_second_moment
    mse = q + b**2
    denominator = 2 * (v + a) + mse
    ccc = np.divide(
        2 * (v + a),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    return {
        "reference_mean": mu,
        "reference_variance": v,
        "disagreement_mean": b,
        "disagreement_variance": q,
        "reference_disagreement_covariance": a,
        "disagreement_mse": mse,
        "ccc": ccc,
        "residual_second_moment": residual_second_moment,
    }


def population_transport(source_moments, target_mean, target_variance):
    """Use the empirical implementation on exact source moments, with no sampling."""
    means, variances = np.broadcast_arrays(target_mean, target_variance)
    predictions = []
    required = [
        "reference_mean",
        "reference_variance",
        "disagreement_mean",
        "disagreement_variance",
        "reference_disagreement_covariance",
        "disagreement_mse",
        "ccc",
    ]
    for j, (mean, variance) in enumerate(zip(means, variances, strict=True)):
        moments = JointMoments(
            **{key: float(source_moments[key][j]) for key in required}
        )
        model = fit_source_model(moments, independent_patients=10)
        predictions.append(
            predict_target(model, ReferenceMoments(float(mean), float(variance)))
        )
    return predictions


def changed_real_law(law: PopulationLaw, battery):
    return replace(
        law,
        bias=law.bias + 0.5 * battery.changed_bias.to_numpy(),
        slope=law.slope + 0.25 * np.tanh(battery.changed_slope.to_numpy()),
        residual_sd=law.residual_sd * np.exp(0.5 * battery.changed_scale.to_numpy()),
    )


def noisy_reference_moments(anatomical_variance):
    """The latent truth exists only here; observed changes use the noisy reference."""
    variance = np.asarray(anatomical_variance, dtype=float)
    if np.any(variance < 0):
        raise ValueError("Anatomical variance cannot be negative")
    return pd.DataFrame(
        {
            "anatomical_variance": variance,
            "reference_variance": variance + 0.3**2,
            "real_observed_disagreement_mse": np.full_like(variance, 0.3**2 + 0.5**2),
            "synthetic_observed_disagreement_mse": np.full_like(variance, 0.5**2),
            "real_latent_error_mse": np.full_like(variance, 0.5**2),
            "synthetic_latent_error_mse": np.full_like(variance, 0.3**2 + 0.5**2),
            "real_reference_disagreement_covariance": np.full_like(variance, -(0.3**2)),
            "synthetic_reference_disagreement_covariance": np.zeros_like(variance),
            "real_ccc": 2 * variance / (2 * variance + 0.3**2 + 0.5**2),
            "synthetic_ccc": 2
            * (variance + 0.3**2)
            / (2 * (variance + 0.3**2) + 0.5**2),
        }
    )


def coverage_fixture(law: PopulationLaw, descriptor_ids):
    """Exact equal/positive/negative truths without searching seeds or near ties."""
    ids = tuple(descriptor_ids)
    if len(ids) < 4 or len(ids) != len(law.bias) or len(set(ids)) != len(ids):
        raise ValueError("Coverage fixtures require at least four distinct descriptors")
    k = max(1, len(ids) // 4)
    arrays = {}
    for field in ["source_mean", "source_variance", "bias", "slope", "residual_sd"]:
        values = np.array(getattr(law, field), copy=True)
        for group in range(1, 4):
            values[group * k : (group + 1) * k] = values[:k]
        arrays[field] = values
    arrays["residual_sd"][2 * k : 3 * k] *= 0.5
    arrays["residual_sd"][3 * k : 4 * k] *= 2.0
    fixture = PopulationLaw(
        **arrays, heteroscedastic_log_sd_slope=law.heteroscedastic_log_sd_slope
    )
    sets = {
        name: ids[g * k : (g + 1) * k]
        for g, name in enumerate(["first", "second", "higher", "lower"])
    }
    truth = population_moments(fixture)["ccc"]
    utilities = {
        name: float(np.mean(truth[g * k : (g + 1) * k])) for g, name in enumerate(sets)
    }
    if not utilities["higher"] > utilities["first"] > utilities["lower"]:
        raise ValueError(
            "This fixture requires positive CCC numerator and nonzero residual variance"
        )
    return fixture, sets, utilities


def hierarchical_draws(
    cfg,
    template: Template,
    law_real,
    law_synthetic,
    *,
    stream,
    replicate,
    draw_count=100,
    target_mean=None,
    target_variance=None,
):
    """Generate the declared correlated fields on supplied metadata, without imputation.

    Missingness flags must be applied by a caller using an audited template. This
    function does not manufacture a missingness mechanism or infer contour counts.
    """
    template.validate()
    if draw_count < 1:
        raise ValueError("At least one synthetic draw is required")
    if len(law_real.bias) != len(template.descriptor_ids) or len(
        law_synthetic.bias
    ) != len(template.descriptor_ids):
        raise ValueError("Law and descriptor template sizes disagree")
    patient_ids = sorted(set(template.patients))
    family_ids = sorted(set(template.families))
    patient_index = np.asarray([patient_ids.index(p) for p in template.patients])
    family_index = np.asarray([family_ids.index(f) for f in template.families])
    p, u, f, j = (
        len(patient_ids),
        len(template.patients),
        len(family_ids),
        len(template.descriptor_ids),
    )
    key = ("patient_fields", template.identity, int(replicate))

    def normal(kind, shape):
        return rng(cfg, stream, *key, kind).normal(size=shape)

    z_x = (
        np.sqrt(0.5)
        * normal("reference_patient_family", (p, f))[patient_index][:, family_index]
        + 0.5 * normal("reference_nodule_family", (u, f))[:, family_index]
        + 0.5 * normal("reference_nodule_descriptor", (u, j))
    )
    mu = law_real.source_mean if target_mean is None else np.asarray(target_mean)
    variance = (
        law_real.source_variance
        if target_variance is None
        else np.asarray(target_variance)
    )
    x = mu + np.sqrt(variance) * z_x
    real_unit = np.repeat(np.arange(u), template.alternate_counts)
    synthetic_unit = np.repeat(np.arange(u), draw_count)

    def alternate(name, law, unit_index):
        n = len(unit_index)
        z = (
            0.5
            * normal(name + "_patient_family", (p, f))[patient_index[unit_index]][
                :, family_index
            ]
            + 0.5 * normal(name + "_nodule_measurement_family", (n, f))[:, family_index]
            + np.sqrt(0.5) * normal(name + "_nodule_measurement_descriptor", (n, j))
        )
        reference = x[unit_index]
        centered = reference - law.source_mean
        residual_sd = law.residual_sd * np.exp(
            law.heteroscedastic_log_sd_slope * centered / np.sqrt(law.source_variance)
        )
        return reference + law.bias + law.slope * centered + residual_sd * z

    real = alternate("real", law_real, real_unit)
    synthetic = alternate("synthetic", law_synthetic, synthetic_unit).reshape(
        u, draw_count, j
    )
    nodule_counts = np.bincount(patient_index, minlength=p)
    unit_weights = 1.0 / (p * nodule_counts[patient_index])
    real_weights = (
        unit_weights[real_unit] / np.asarray(template.alternate_counts)[real_unit]
    )
    return {
        "reference": x,
        "real": real,
        "synthetic": synthetic,
        "real_unit_index": real_unit,
        "unit_weights": unit_weights,
        "real_weights": real_weights,
        "patient_index": patient_index,
    }


def demonstration_template():
    """A declared artificial numerical fixture, never an empirical cohort substitute."""
    return Template(
        "fixed_demonstration_v1",
        tuple(f"toy_{j:02d}" for j in range(12)),
        tuple(f"family_{j // 4}" for j in range(12)),
        tuple(f"patient_{i:02d}" for i in range(24)),
        tuple("unit" for _ in range(24)),
        tuple(2 for _ in range(24)),
    )
