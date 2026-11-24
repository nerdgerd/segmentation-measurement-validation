"""Return simulation tables for an explicitly supplied battery and declared streams.

The caller owns execution locks and manifests. Population calculations are exact
Gaussian moments; artificial smoke tables never stand in for audited cohort-shaped
finite-sample or interval-coverage experiments.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .moments import weighted_joint_moments
from .simulation import (
    coefficient_battery,
    make_laws,
    population_moments,
    population_transport,
    changed_real_law,
    noisy_reference_moments,
    hierarchical_draws,
    demonstration_template,
)


def _association(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _selection(ids, scores):
    k = max(1, len(ids) // 4)
    return sorted(range(len(ids)), key=lambda j: (-float(scores[j]), str(ids[j])))[:k]


def population_tables(
    cfg,
    descriptor_ids,
    families,
    *,
    template_key,
    template_role="audited_structure",
    stream="mechanism_final",
):
    battery = coefficient_battery(
        cfg, descriptor_ids, families, template_key=template_key, stream=stream
    )
    spec = cfg["simulation"]
    mechanism_rows, profile_rows, remedy_rows, utility_rows = [], [], [], []
    ids = battery.descriptor_id.to_list()
    for regime, theta, h in product(
        spec["core_regimes"],
        spec["population_theta_grid"],
        spec["population_reference_log_dispersion_multipliers"],
    ):
        real, syn = make_laws(battery, h=h, theta=theta, regime=regime)
        r, s = population_moments(real), population_moments(syn)
        ratio = 0.5 * np.log2(s["disagreement_mse"] / r["disagreement_mse"])
        keys = {
            "template": template_key,
            "template_role": template_role,
            "regime": regime,
            "theta": theta,
            "h": h,
        }
        for j, descriptor in enumerate(ids):
            mechanism_rows.append(
                {
                    **keys,
                    "descriptor_id": descriptor,
                    "family": families[j],
                    **{f"real_{key}": value[j] for key, value in r.items()},
                    **{f"synthetic_{key}": value[j] for key, value in s.items()},
                    "log2_rmse_ratio": ratio[j],
                }
            )
        profile_rows.append(
            {
                **keys,
                "descriptor_count": len(ids),
                "population_profile_spearman": _association(s["ccc"], r["ccc"]),
                "realized_log_scale_pearson": float(
                    np.corrcoef(np.log(real.residual_sd), np.log(syn.residual_sd))[0, 1]
                ),
                "median_signed_log2_rmse_ratio": float(np.median(ratio)),
                "median_absolute_log2_rmse_ratio": float(np.median(np.abs(ratio))),
            }
        )

    # The remedy has a separate coefficient stream, shared across its case-mix interventions.
    remedy_battery = coefficient_battery(
        cfg,
        descriptor_ids,
        families,
        template_key=template_key,
        stream=("smoke" if stream == "smoke" else "remedy_final"),
    )
    for regime, theta in product(spec["core_regimes"], spec["population_theta_grid"]):
        real, syn = make_laws(remedy_battery, h=1.0, theta=theta, regime=regime)
        source = population_moments(real)
        conditions = [
            ("unchanged_conditional_law", r, shift, real)
            for r, shift in product(
                spec["remedy_target_reference_variance_multipliers"],
                spec["remedy_target_mean_shift_source_sd"],
            )
        ]
        if regime == "biased_proportional":
            conditions.append(
                (
                    "changed_conditional_law",
                    2.0,
                    1.0,
                    changed_real_law(real, remedy_battery),
                )
            )
        for scenario, multiplier, shift_sd, target_law in conditions:
            target_mu = real.source_mean + shift_sd * np.sqrt(real.source_variance)
            target_v = multiplier * real.source_variance
            truth = population_moments(target_law, target_mu, target_v)
            synthetic = population_moments(syn, target_mu, target_v)
            predictions = population_transport(source, target_mu, target_v)
            predicted = np.asarray([p.ccc for p in predictions])
            scores = {
                "source_ccc": source["ccc"],
                "synthetic_ccc": synthetic["ccc"],
                "error_transport": predicted,
            }
            keys = {
                "template": template_key,
                "template_role": template_role,
                "regime": regime,
                "theta": theta,
                "h": 1.0,
                "scenario": scenario,
                "target_variance_multiplier": multiplier,
                "target_mean_shift_source_sd": shift_sd,
            }
            for j, descriptor in enumerate(ids):
                remedy_rows.append(
                    {
                        **keys,
                        "descriptor_id": descriptor,
                        "family": families[j],
                        "target_population_ccc": truth["ccc"][j],
                        "target_population_disagreement_mse": truth["disagreement_mse"][
                            j
                        ],
                        "transport_disagreement_mse": predictions[j].disagreement_mse,
                        **{
                            f"{name}_predicted_ccc": score[j]
                            for name, score in scores.items()
                        },
                        **{
                            f"{name}_absolute_ccc_error": abs(
                                score[j] - truth["ccc"][j]
                            )
                            for name, score in scores.items()
                        },
                    }
                )
            utility = {
                name: float(np.mean(truth["ccc"][_selection(ids, score)]))
                for name, score in scores.items()
            }
            for name, score in scores.items():
                selected = _selection(ids, score)
                utility_rows.append(
                    {
                        **keys,
                        "policy": name,
                        "J": len(ids),
                        "k": len(selected),
                        "selected_ids": "|".join(ids[j] for j in selected),
                        "population_selected_set_mean_ccc": utility[name],
                        "mean_absolute_ccc_prediction_error": float(
                            np.mean(np.abs(score - truth["ccc"]))
                        ),
                        "error_transport_minus_source_ccc": utility["error_transport"]
                        - utility["source_ccc"],
                        "synthetic_ccc_minus_error_transport": utility["synthetic_ccc"]
                        - utility["error_transport"],
                        "synthetic_ccc_minus_source_ccc": utility["synthetic_ccc"]
                        - utility["source_ccc"],
                    }
                )
    noisy = noisy_reference_moments(np.exp(2 * remedy_battery.reference.to_numpy()))
    noisy["descriptor_id"], noisy["template"], noisy["template_role"] = (
        ids,
        template_key,
        template_role,
    )
    return {
        "population_mechanism": pd.DataFrame(mechanism_rows),
        "population_mechanism_summary": pd.DataFrame(profile_rows),
        "transport_stress_tests": pd.DataFrame(remedy_rows),
        "population_policy_utility": pd.DataFrame(utility_rows),
        "noisy_reference_interpretation": noisy,
        "mechanism_coefficient_fields": battery,
        "remedy_coefficient_fields": remedy_battery,
    }


def smoke_tables(cfg, *, replicates=100):
    """Execute numerical development repeats, always labeled nonfinal and artificial."""
    template = demonstration_template()
    battery = coefficient_battery(
        cfg,
        template.descriptor_ids,
        template.families,
        template_key=template.name,
        stream="smoke",
    )
    rows = []
    for regime in cfg["simulation"]["core_regimes"]:
        real, syn = make_laws(battery, h=1.0, theta=0.5, regime=regime)
        truth_r, truth_s = population_moments(real), population_moments(syn)
        for replicate in range(replicates):
            values = hierarchical_draws(
                cfg,
                template,
                real,
                syn,
                stream="smoke",
                replicate=replicate,
                draw_count=100,
            )
            x = values["reference"]
            real_x = x[values["real_unit_index"]]
            single_x = np.repeat(x, 100, axis=0)
            synthetic = values["synthetic"].reshape(-1, len(template.descriptor_ids))
            synthetic_weights = np.repeat(values["unit_weights"] / 100, 100)
            for j, descriptor in enumerate(template.descriptor_ids):
                mr = weighted_joint_moments(
                    real_x[:, j], values["real"][:, j], values["real_weights"]
                )
                ms = weighted_joint_moments(
                    single_x[:, j], synthetic[:, j], synthetic_weights
                )
                rows.append(
                    {
                        "template": template.name,
                        "template_role": "artificial_numerical_smoke",
                        "final_simulation_estimate": False,
                        "regime": regime,
                        "replicate": replicate,
                        "descriptor_id": descriptor,
                        "real_sample_ccc": mr.ccc,
                        "real_population_ccc": truth_r["ccc"][j],
                        "synthetic_sample_ccc": ms.ccc,
                        "synthetic_population_ccc": truth_s["ccc"][j],
                    }
                )
    return {"numerical_smoke": pd.DataFrame(rows)}
