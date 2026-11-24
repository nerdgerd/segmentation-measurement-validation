"""Construction uses source alternates and physically separated target references."""

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from redesign2.construction import (
    construct_direction,
    construct_from_frames,
    patient_cache_from_frame,
    source_pair_matrix,
    synthetic_matrix,
    reference_matrix,
)
from redesign2.empirical_math import aggregate_patient_moments
from redesign2.provenance import content_hash, file_hash, read_protocol, verify_run


def fixture_inputs(draw_count=100):
    cfg = read_protocol(Path(__file__).resolve().parents[1] / "config/protocol.yaml")
    descriptors = tuple("abcdefgh")

    def measured(cohort, role):
        rows = []
        for patient in range(11):
            for feature, descriptor in enumerate(descriptors):
                x = (1 + 0.1 * feature) * (patient - 5) + feature
                if cohort == "brats_tcga":
                    x = 1.3 * x + 0.2 * feature
                value = (
                    x
                    if role == "reference"
                    else x + 0.1 * feature + 0.15 * ((patient % 3) - 1)
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "patient_id": f"p{patient:02}",
                        "unit_id": "unit",
                        "annotation_id": "ref" if role == "reference" else "alt",
                        "role": role,
                        "descriptor_id": descriptor,
                        "value": value,
                        "definedness": "defined",
                    }
                )
        return pd.DataFrame(rows)

    source_reference = measured("upenn", "reference")
    source_alternate = measured("upenn", "alternate")
    target_reference = measured("brats_tcga", "reference")
    reference_values = target_reference.pivot(
        index="patient_id", columns="descriptor_id", values="value"
    )
    values, flags = [], []
    for patient in reference_values.index:
        for draw in range(draw_count):
            row = {
                "cohort": "brats_tcga",
                "patient_id": patient,
                "unit_id": "unit",
                "draw_id": draw,
                "batch": draw // 20,
            }
            flag = row.copy()
            for feature, descriptor in enumerate(descriptors):
                row[descriptor] = reference_values.loc[patient, descriptor] + (
                    0.4 + feature * 0.1
                ) * (-1 if draw % 2 else 1)
                flag[descriptor] = "defined"
            values.append(row)
            flags.append(flag)
    frames = {
        "source_reference": source_reference,
        "source_alternate": source_alternate,
        "target_reference": target_reference,
        "draw_features": pd.DataFrame(values),
        "draw_definedness": pd.DataFrame(flags),
    }
    return cfg, descriptors, frames


def construct(cfg, descriptors, frames, scope="full"):
    return construct_from_frames(
        cfg,
        source="upenn",
        target="brats_tcga",
        scope=scope,
        descriptor_ids=descriptors,
        **frames,
    )


def test_all_three_constructions_fit_source_once_and_keep_target_alternates_out_of_api():
    cfg, descriptors, frames = fixture_inputs()
    result = construct(cfg, descriptors, frames)
    assert result.ready
    assert result.universe == descriptors
    assert all(selection.budget == 2 for selection in result.selections)
    assert set(result.tables["construction_scores"].policy) == {
        "source_ccc",
        "synthetic_ccc",
        "error_transport",
    }
    assert not result.tables["draw_convergence_summary"].refinement_required.iloc[0]
    assert "target_alternate" not in inspect.signature(construct_from_frames).parameters
    discovery = {name: frame.copy() for name, frame in frames.items()}
    for name in ("target_reference", "draw_features", "draw_definedness"):
        discovery[name] = discovery[name].loc[
            discovery[name].patient_id.isin([f"p{i:02}" for i in range(5)])
        ]
    audit_screen = construct(cfg, descriptors, discovery, scope="discovery")
    assert audit_screen.ready
    pd.testing.assert_frame_equal(
        result.tables["source_models"].drop(columns="scope"),
        audit_screen.tables["source_models"].drop(columns="scope"),
    )


def test_source_definedness_gates_common_universe_without_filling_a_missing_score():
    cfg, descriptors, frames = fixture_inputs()
    missing = frames["source_alternate"].descriptor_id.eq("h") & frames[
        "source_alternate"
    ].patient_id.isin(["p00", "p01"])
    frames["source_alternate"].loc[missing, "value"] = np.nan
    frames["source_alternate"].loc[missing, "definedness"] = "structural_absence"
    result = construct(cfg, descriptors, frames)
    assert result.ready
    assert result.universe == descriptors[:-1]
    assert all(selection.budget == 1 for selection in result.selections)
    excluded = result.tables["construction_scores"].query("descriptor_id == 'h'")
    assert not excluded.eligible.any()
    assert excluded.eligibility_reason.str.contains(
        "insufficient_independent_patients"
    ).all()


def test_patient_cache_roundtrip_preserves_declared_subset_and_rejects_duplicate_identity():
    cfg, descriptors, frames = fixture_inputs()
    result = construct(cfg, descriptors, frames)
    cached = result.tables["synthetic_patient_moments"]
    decoded = patient_cache_from_frame(cached, descriptor_ids=["g", "a"])
    profile = aggregate_patient_moments(decoded)
    originals = result.tables["synthetic_moments"].set_index("descriptor_id")
    assert decoded.descriptor_ids == ("g", "a")
    assert [moment.ccc for moment in profile.moments] == pytest.approx(
        originals.loc[["g", "a"], "ccc"]
    )
    assert not decoded.reference_mean.flags.writeable
    with pytest.raises(ValueError, match="Duplicate"):
        patient_cache_from_frame(
            pd.concat([cached, cached.iloc[:1]]), descriptor_ids=descriptors
        )


def test_draw_precision_failure_requires_one_direction_refinement_before_selection():
    cfg, descriptors, frames = fixture_inputs()
    later = frames["draw_features"].draw_id >= 50
    for descriptor in descriptors:
        frames["draw_features"].loc[later, descriptor] += 5
    result = construct(cfg, descriptors, frames)
    assert not result.ready
    assert result.reason == "draw_refinement_required"
    assert result.selections == ()
    assert result.tables["selections"].empty
    assert result.tables["draw_convergence_summary"].refinement_required.iloc[0]
    assert not result.tables["construction_scores"].selected.any()


def test_remaining_precision_failure_after200_is_retained_without_third_draw_search():
    cfg, descriptors, frames = fixture_inputs(draw_count=200)
    later = frames["draw_features"].draw_id >= 100
    for descriptor in descriptors:
        frames["draw_features"].loc[later, descriptor] += 8
    result = construct(cfg, descriptors, frames)
    assert result.ready
    summary = result.tables["draw_convergence_summary"].iloc[0]
    assert summary.precision_unresolved_after_allowed_refinement
    assert not summary.refinement_required
    assert len(result.selections) == 3


def test_reference_role_mismatch_and_reader_duplicated_draws_are_rejected():
    cfg, descriptors, frames = fixture_inputs()
    wrong = frames["source_alternate"].copy()
    wrong["cohort"] = "brats_tcga"
    with pytest.raises(PermissionError, match="cohort or contour role"):
        source_pair_matrix(
            frames["source_reference"],
            wrong,
            cohort="upenn",
            descriptor_ids=descriptors,
        )
    target = reference_matrix(
        frames["target_reference"], cohort="brats_tcga", descriptor_ids=descriptors
    )
    duplicated = pd.concat([frames["draw_features"], frames["draw_features"].iloc[:1]])
    with pytest.raises(ValueError, match="Repeated synthetic draw"):
        synthetic_matrix(
            target, duplicated, frames["draw_definedness"], cohort="brats_tcga"
        )


def test_discovery_file_capabilities_never_open_full_target_reference_or_alternate(
    tmp_path, monkeypatch
):
    cfg, descriptors, frames = fixture_inputs()
    prepared, generated = tmp_path / "prepared", tmp_path / "generated"
    generated.mkdir()
    paths = {}
    for cohort, role, name in (
        ("upenn", "reference", "source_reference"),
        ("upenn", "alternate", "source_alternate"),
        ("brats_tcga", "reference", "target_reference"),
    ):
        (prepared / cohort).mkdir(parents=True, exist_ok=True)
        file = prepared / cohort / (role + "_features.csv")
        frames[name].to_csv(file, index=False)
        paths[str(file.relative_to(prepared)).replace("\\", "/")] = file_hash(file)
    discovery_patients = [f"p{i:02}" for i in range(5)]
    discovery_reference = prepared / "brats_tcga/discovery_reference_features.csv"
    frames["target_reference"].loc[
        frames["target_reference"].patient_id.isin(discovery_patients)
    ].to_csv(discovery_reference, index=False)
    paths["brats_tcga/discovery_reference_features.csv"] = file_hash(
        discovery_reference
    )
    poison = prepared / "brats_tcga/alternate_features.csv"
    poison.write_text("THIS FILE MUST NOT BE OPENED")
    paths["brats_tcga/alternate_features.csv"] = file_hash(poison)
    (prepared / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": content_hash(cfg),
                "status": "complete",
                "outputs": paths,
            }
        )
    )
    generated_paths = {}
    for name in ("draw_features", "draw_definedness"):
        path = generated / ("discovery_" + name + ".csv")
        frames[name].loc[frames[name].patient_id.isin(discovery_patients)].to_csv(
            path, index=False
        )
        generated_paths[path.name] = file_hash(path)
    (generated / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": content_hash(cfg),
                "status": "complete",
                "outputs": generated_paths,
                "extra": {"source": "upenn", "target": "brats_tcga", "draw_count": 100},
            }
        )
    )
    descriptor_file = tmp_path / "descriptors.csv"
    pd.DataFrame(
        {
            "feature_id": descriptors,
            "domain": "MRI",
            "family": "shape",
            "region": "WT",
            "units": "unit",
            "definition": "fixed toy descriptor",
        }
    ).to_csv(descriptor_file, index=False)
    monkeypatch.setattr(
        "redesign2.provenance.code_state", lambda workspace: {"test": True}
    )
    original_open = Path.open
    forbidden = {
        poison.resolve(),
        (prepared / "brats_tcga/reference_features.csv").resolve(),
    }

    def guarded_open(path, *args, **kwargs):
        if path.resolve() in forbidden:
            raise AssertionError("Forbidden target alternate or audit reference access")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = construct_direction(
        tmp_path,
        cfg,
        prepared,
        generated,
        source="upenn",
        target="brats_tcga",
        scope="discovery",
        run_id="constructed",
        descriptor_manifest=descriptor_file,
    )
    status = json.loads((result / "status.json").read_text())
    assert status["status"] == "construction_complete"
    assert verify_run(result) == []
    access = json.loads((result / "construction_status.json").read_text())[
        "access_record"
    ]
    assert {entry["name"] for entry in access} == {
        "source_reference",
        "source_alternate",
        "target_reference",
        "draw_features",
        "draw_definedness",
    }
