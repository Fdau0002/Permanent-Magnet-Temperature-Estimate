"""Synthetic evidence exercises the real exporter/importer, never scientific claims."""

import copy
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
import pytest

from pmsm_project.config import EXPECTED_SHA256, LOCKED_DEVELOPMENT_IDS
from pmsm_project.search import MODEL_SPACES, SPAN_SPACE, decode_spans
from pmsm_project.handoff import (
    export_manifest,
    load_manifest,
    configuration_agreement,
    select_winner,
    _seal,
    start_development_run,
    write_handoff_archive,
)
from pmsm_project.models import build_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    """Complete fabricated tables ONLY for validating the IO contract."""

    def trials(count, name=None, start=1):
        rows = []
        for i in range(count):
            params = [20 + i, 40, 200, 800]
            hp = {}
            if name:
                values = {
                    "SVR": [0.2, 0.1],
                    "k-NN": [12, "distance"],
                    "Random Forest": [20, 6, 4, 2, "yes"],
                    "Extra Trees": [20, 6, 4, 2, "no"],
                    "MLP": [1, 4, 0.001, 0.003],
                }[name]
                params += values
                hp = {d.name: v for d, v in zip(MODEL_SPACES[name], values)}
            spans = decode_spans(params[:4])
            folds = [20.0 - i / 10, 21.0 - i / 10, 22.0 - i / 10]
            rows.append(
                {
                    "trial": i + start,
                    "parameters_json": json.dumps(params),
                    "hyperparameters_json": json.dumps(hp),
                    **{f"s{j+1}": v for j, v in enumerate(spans)},
                    **{f"fold{j+1}_mse": v for j, v in enumerate(folds)},
                    "mean_cv_mse": np.mean(folds),
                }
            )
        return pd.DataFrame(rows)

    tables = {
        "ols_pilot": trials(8),
        "ols_expanded": trials(20, start=9),
        "ols_full": trials(20),
    }
    for name in MODEL_SPACES:
        tables[f"pilot_{name}"] = trials(8, name)
        tables[f"scaled_{name}"] = trials(12, name)
    tables["svr_nested"] = pd.DataFrame(
        [
            {"Samples_per_Profile": n, "Fold": f, "Repeat": r, "MSE": 20.0}
            for n in [80, 160, 320, 640]
            for f in [1, 2, 3]
            for r in [1, 2, 3]
        ]
    )
    for name, n in [("stage9h", 6), ("stage9i", 15)]:
        tables[name] = pd.DataFrame({"MSE_C2": np.arange(n) + 20.0})
    cs = [0.01, 0.02, 0.04, 0.08]
    for name, values in [("stage9j", cs[2:]), ("stage9k", cs[:2])]:
        tables[name] = pd.DataFrame(
            [
                {
                    "Fold": f,
                    "C": c,
                    "Training_Rows": (36 if f == 1 else 37) * 1280,
                    "MSE_C2": 20 + c,
                }
                for f in [1, 2, 3]
                for c in values
            ]
        )
    tables["stage9k_summary"] = pd.DataFrame(
        {"C": cs, "Mean_MSE_C2": [22.0, 20.0, 21.0, 23.0], "Warnings": [0, 0, 0, 0]}
    )
    cv = pd.DataFrame(
        {
            "profile_id": LOCKED_DEVELOPMENT_IDS,
            "cv_fold": [1] * 19 + [2] * 18 + [3] * 18,
        }
    )
    return tables, cv


def publish(tmp_path, evidence):
    tables, cv = evidence
    path = tmp_path / "development_selected_configurations.json"
    manifest = export_manifest(
        path,
        project_root=ROOT,
        run_id=str(uuid4()),
        started_utc="2026-10-25T00:00:00+00:00",
        dataset_hash=EXPECTED_SHA256,
        profile_cv=cv,
        tables=tables,
    )
    return path, manifest, cv


def test_real_export_import_and_six_constructors(tmp_path, evidence):
    path, written, cv = publish(tmp_path, evidence)
    loaded = load_manifest(
        path, project_root=ROOT, dataset_hash=EXPECTED_SHA256, profile_cv=cv
    )
    assert loaded == written
    mapping = copy.deepcopy(loaded["models"])
    agreement = configuration_agreement(loaded, mapping)
    assert len(agreement) == 6 and agreement.Agrees.all()
    assert mapping["OLS"]["spans"] == [
        39,
        79,
        279,
        1079,
    ]  # fixture minimum, not historical OLS
    assert mapping["LinearSVR"]["spans"] == loaded["svr_reference"]["spans"]
    adjustment = mapping["LinearSVR"]["regularisation_adjustment"]
    assert mapping["LinearSVR"]["hyperparameters"]["C"] * 1055198 == pytest.approx(
        adjustment["reference_C"] * adjustment["mean_cv_training_rows"]
    )
    for name in mapping:
        if name != "OLS":
            assert hasattr(build_pipeline(name, mapping), "predict")
    mapping["MLP"]["spans"][0] += 1
    with pytest.raises(ValueError, match="diverge"):
        configuration_agreement(loaded, mapping)


def test_final_factory_has_no_fallback():
    with pytest.raises(TypeError):
        build_pipeline("MLP")


def test_missing_manifest_fails_with_handoff_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run Notebook 02"):
        load_manifest(
            tmp_path / "absent.json", project_root=ROOT, dataset_hash=EXPECTED_SHA256
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "smoke",
        "incomplete",
        "dataset",
        "profiles",
        "software",
        "budget",
        "code",
        "model",
        "missing_model",
        "evidence",
        "checksum",
        "folds",
    ],
)
def test_incompatible_manifest_refused(tmp_path, evidence, mutation):
    path, payload, cv = publish(tmp_path, evidence)
    if mutation == "schema":
        payload["schema_version"] = "1.0"
    elif mutation == "smoke":
        payload["run"]["mode"] = "smoke"
    elif mutation == "incomplete":
        payload["run"]["status"] = "running"
    elif mutation == "dataset":
        payload["dataset"]["sha256"] = "0" * 64
    elif mutation == "profiles":
        payload["dataset"]["development_profile_ids"] = []
    elif mutation == "software":
        payload["software"]["scikit-learn"] = "0.0"
    elif mutation == "budget":
        payload["search_protocol"]["scaled_models"]["trials_each"] = 5
    elif mutation == "code":
        payload["methodology_sha256"] = "0" * 64
    elif mutation == "model":
        payload["models"]["MLP"]["hyperparameters"]["n_units"] = 99
    elif mutation == "missing_model":
        del payload["models"]["k-NN"]
    elif mutation == "evidence":
        payload["evidence"].pop("stage9i")
    elif mutation == "folds":
        payload["validation"]["fold_by_profile"]["2"] = 3
    elif mutation == "checksum":
        payload["models"]["OLS"]["spans"][0] += 1
    if mutation != "checksum":
        payload["manifest_sha256"] = _seal(payload)
    path.write_text(json.dumps(payload))
    if mutation != "checksum":
        marker = path.with_name("development_run_status.json")
        status = json.loads(marker.read_text())
        status["manifest_sha256"] = payload["manifest_sha256"]
        marker.write_text(json.dumps(status))
    with pytest.raises(ValueError):
        load_manifest(
            path, project_root=ROOT, dataset_hash=EXPECTED_SHA256, profile_cv=cv
        )


def test_missing_and_altered_evidence_refused(tmp_path, evidence):
    path, payload, cv = publish(tmp_path, evidence)
    target = path.parent / payload["evidence"]["ols_full"]["path"]
    original = target.read_bytes()
    target.unlink()
    with pytest.raises(FileNotFoundError):
        load_manifest(path, project_root=ROOT, dataset_hash=EXPECTED_SHA256)
    target.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="evidence changed"):
        load_manifest(path, project_root=ROOT, dataset_hash=EXPECTED_SHA256)


@pytest.mark.parametrize(
    "fault",
    [
        "smoke",
        "partial",
        "nan",
        "wrong_mean",
        "outside_space",
        "missing_stage",
        "bad_rows",
        "convergence",
    ],
)
def test_export_refuses_invalid_evidence(tmp_path, evidence, fault):
    tables, cv = evidence
    mode = "full"
    if fault == "smoke":
        mode = "smoke"
    elif fault == "partial":
        tables["scaled_SVR"] = tables["scaled_SVR"].iloc[:-1]
    elif fault == "nan":
        tables["ols_full"].loc[0, "mean_cv_mse"] = np.nan
    elif fault == "wrong_mean":
        tables["ols_full"].loc[0, "mean_cv_mse"] = -1
    elif fault == "outside_space":
        tables["scaled_SVR"].loc[0, "parameters_json"] = "[999,40,200,800,0.2,0.1]"
    elif fault == "missing_stage":
        tables.pop("stage9h")
    elif fault == "bad_rows":
        tables["stage9k"]["Training_Rows"] = 100
    elif fault == "convergence":
        tables["stage9k_summary"]["Warnings"] = 3
    with pytest.raises(ValueError):
        export_manifest(
            tmp_path / "none.json",
            project_root=ROOT,
            run_id=str(uuid4()),
            started_utc="test",
            dataset_hash=EXPECTED_SHA256,
            profile_cv=cv,
            tables=tables,
            mode=mode,
        )
    assert not (tmp_path / "none.json").exists()


def test_model_winner_changes_with_current_scores(evidence):
    tables, _ = evidence
    t = tables["scaled_SVR"]
    old = select_winner(t, 12)
    t.loc[0, ["fold1_mse", "fold2_mse", "fold3_mse", "mean_cv_mse"]] = 1.0
    new = select_winner(t, 12)
    assert new["selection"]["trial"] == 1 and new["spans"] != old["spans"]


def test_new_incomplete_run_blocks_old_completed_manifest(tmp_path, evidence):
    path, payload, cv = publish(tmp_path, evidence)
    start_development_run(path, str(uuid4()), "2026-10-25T01:00:00+00:00")
    with pytest.raises(ValueError, match="incomplete or superseded"):
        load_manifest(
            path, project_root=ROOT, dataset_hash=EXPECTED_SHA256, profile_cv=cv
        )
    with pytest.raises(ValueError, match="incomplete or superseded"):
        write_handoff_archive(path)


def test_complete_archive_transfers_to_an_independent_directory(tmp_path, evidence):
    import zipfile

    path, manifest, cv = publish(tmp_path, evidence)
    archive = write_handoff_archive(path)
    other = tmp_path / "new_runtime"
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert len(zipped.namelist()) == len(manifest["evidence"]) + 2
        zipped.extractall(other)
    loaded = load_manifest(
        other / path.name,
        project_root=ROOT,
        dataset_hash=EXPECTED_SHA256,
        profile_cv=cv,
    )
    assert loaded == manifest
    marker = other / "development_run_status.json"
    marker.unlink()
    with pytest.raises(FileNotFoundError, match="run-status"):
        load_manifest(
            other / path.name, project_root=ROOT, dataset_hash=EXPECTED_SHA256
        )


def test_adjustment_metadata_and_selected_fold_scores_cannot_diverge(
    tmp_path, evidence
):
    for key in ["adjustment", "fold_score"]:
        path, payload, cv = publish(tmp_path, evidence)
        if key == "adjustment":
            payload["models"]["LinearSVR"]["regularisation_adjustment"][
                "reference_C"
            ] *= 2
        else:
            payload["models"]["OLS"]["selection"]["fold_mse_c2"][0] += 1
        payload["manifest_sha256"] = _seal(payload)
        path.write_text(json.dumps(payload))
        marker = path.with_name("development_run_status.json")
        marker.write_text(
            json.dumps(
                {
                    "run_id": payload["run"]["id"],
                    "status": "completed",
                    "manifest_sha256": payload["manifest_sha256"],
                }
            )
        )
        with pytest.raises(ValueError):
            load_manifest(
                path, project_root=ROOT, dataset_hash=EXPECTED_SHA256, profile_cv=cv
            )
