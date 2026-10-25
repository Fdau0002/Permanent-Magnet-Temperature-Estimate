"""Fail-closed handoff from development selection to final evaluation.

Manifests are evidence records, not a source of default hyperparameters.
Production export requires complete trial tables from the current run.
SHA-256 detects accidental mismatch; it is not a digital signature.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from uuid import UUID
import zipfile

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_SHA256,
    LOCKED_DEVELOPMENT_IDS,
    LOCKED_TEST_IDS,
    TRAIN_ORDER,
)
from .data import file_sha256
from .search import protocol

SCHEMA_VERSION = "2.0"
FEATURE_CONTRACT = {
    "inputs": 12,
    "features": 108,
    "sample_frequency_hz": 2.0,
    "adjust": True,
    "ewms_bias": True,
    "profile_reset": True,
    "history_before_sampling": True,
    "order": "variable-major",
    "input_scaler": "StandardScaler fitted to training partition only",
    "target_scaler": "StandardScaler except QR OLS; inverse transform before scoring",
}


def software_versions() -> dict:
    """Return Python and numerical-library versions for strict compatibility.

    Returns
    -------
    dict
        Python major/minor plus exact installed package versions.

    Raises
    ------
    importlib.metadata.PackageNotFoundError
        If a required numerical dependency is absent.
    """
    result = {"python": ".".join(platform.python_version_tuple()[:2])}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-optimize",
        "joblib",
    ]:
        result[name] = importlib.metadata.version(name)
    return result


def methodology_digest(project_root: str | Path) -> str:
    """Hash production source and notebook code, excluding outputs and Markdown.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Extracted project containing ``src`` and both numbered notebooks.

    Returns
    -------
    str
        SHA-256 over path-qualified code bytes in deterministic order.

    Raises
    ------
    OSError, ValueError
        If required source notebooks cannot be read.
    """
    root = Path(project_root)
    digest = hashlib.sha256()
    for path in sorted((root / "src/pmsm_project").glob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    for name in [
        "02_Experimental_Development.ipynb",
        "01_Final_Reproducible_Evaluation.ipynb",
    ]:
        notebook = json.loads((root / "notebooks" / name).read_text())
        digest.update(name.encode())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                digest.update("".join(cell["source"]).encode())
    return digest.hexdigest()


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _seal(payload: dict) -> str:
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def start_development_run(path: str | Path, run_id: str, started_utc: str) -> Path:
    """Mark a new run active so a previous successful selection cannot be reused.

    Parameters
    ----------
    path : str or pathlib.Path
        Authoritative selection-manifest destination.
    run_id : str
        New UUID generated for this Run All.
    started_utc : str
        ISO-8601 execution start time.

    Returns
    -------
    pathlib.Path
        Run-status JSON next to the manifest. Historical run evidence is retained.

    Raises
    ------
    ValueError
        If the run identifier is not a UUID.
    OSError
        If the status marker cannot be saved.

    Notes
    -----
    A failed or interrupted new run leaves this marker running. The final
    loader therefore rejects an older manifest even if it was once complete.
    """
    UUID(run_id)
    target = Path(path).resolve().with_name("development_run_status.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(
            {"run_id": run_id, "status": "running", "started_utc": started_utc},
            indent=2,
        )
    )
    temporary.replace(target)
    return target


def _check_evidence(tables):
    required = {
        "ols_pilot": 8,
        "ols_expanded": 20,
        "ols_full": 20,
        "svr_nested": 36,
        "stage9h": 6,
        "stage9i": 15,
        "stage9j": 6,
        "stage9k": 6,
        "stage9k_summary": 4,
    }
    for name in ["SVR", "k-NN", "Random Forest", "Extra Trees", "MLP"]:
        required[f"pilot_{name}"] = 8
        required[f"scaled_{name}"] = 12
    if set(tables) != set(required):
        raise ValueError("Required search and numerical evidence is incomplete")
    for name, count in required.items():
        if len(tables[name]) != count:
            raise ValueError(f"{name} requires {count} evidence records")
    # Validate all search rows, including losers, against the original dimensions.
    from .search import SPAN_SPACE, MODEL_SPACES, encode_spans, decode_spans

    for prefix in ["pilot", "scaled"]:
        for name, dims in MODEL_SPACES.items():
            table = tables[f"{prefix}_{name}"]
            select_winner(table, 8 if prefix == "pilot" else 12)
            for row in table.itertuples():
                parameters = json.loads(row.parameters_json)
                spaces = SPAN_SPACE + dims
                if len(parameters) != len(spaces) or any(
                    value not in dimension
                    for value, dimension in zip(parameters, spaces)
                ):
                    raise ValueError(f"{name}: candidate outside original search space")
                spans = [int(getattr(row, f"s{i}")) for i in (1, 2, 3, 4)]
                if decode_spans(parameters[:4]) != spans:
                    raise ValueError(
                        f"{name}: candidate spans do not match search parameters"
                    )
                hp = json.loads(row.hyperparameters_json)
                if hp != {
                    dimension.name: value
                    for dimension, value in zip(dims, parameters[4:])
                }:
                    raise ValueError(
                        f"{name}: model hyperparameters differ from trial proposal"
                    )
    for name in ["ols_full", "ols_expanded"]:
        for row in tables[name].itertuples():
            spans = [int(getattr(row, f"s{i}")) for i in (1, 2, 3, 4)]
            if any(
                value not in dimension
                for value, dimension in zip(encode_spans(spans), SPAN_SPACE)
            ):
                raise ValueError("OLS candidate outside expanded span search space")
    for name in ["stage9h", "stage9i", "stage9j", "stage9k"]:
        scores = tables[name]["MSE_C2"].to_numpy(float)
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError(f"{name} has invalid numerical evidence")
    nested = tables["svr_nested"]
    if set(zip(nested.Samples_per_Profile, nested.Fold, nested.Repeat)) != {
        (n, f, r) for n in [80, 160, 320, 640] for f in [1, 2, 3] for r in [1, 2, 3]
    }:
        raise ValueError(
            "Nested SVR evidence does not retain all four budgets and three repeats"
        )


def select_winner(trials: pd.DataFrame, expected_trials: int) -> dict:
    """Select the first minimum mean grouped-CV MSE in a complete fresh table.

    Parameters
    ----------
    trials : pandas.DataFrame
        Chronological trial rows with spans, three fold MSE values (deg C
        squared), arithmetic mean and optional hyperparameters JSON.
    expected_trials : int
        Original campaign count; incomplete tables are rejected.

    Returns
    -------
    dict
        Spans, hyperparameters, trial number and supporting fold scores.

    Raises
    ------
    ValueError
        If rows are missing, nonfinite, duplicated, out of sequence or inconsistent.
    """
    required = [
        "trial",
        "mean_cv_mse",
        "fold1_mse",
        "fold2_mse",
        "fold3_mse",
        "s1",
        "s2",
        "s3",
        "s4",
    ]
    if len(trials) != expected_trials or not set(required) <= set(trials):
        raise ValueError(
            f"Expected {expected_trials} complete trial rows and columns {required}"
        )
    if trials.trial.tolist() != list(range(1, expected_trials + 1)):
        raise ValueError("Trial numbers must be unique and chronological from 1")
    numeric = trials[required].to_numpy(dtype=float)
    scores = trials[[f"fold{i}_mse" for i in (1, 2, 3)]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (scores < 0).any():
        raise ValueError("All trial scores must be finite and nonnegative")
    if not np.allclose(scores.mean(axis=1), trials.mean_cv_mse, rtol=1e-12, atol=1e-12):
        raise ValueError("Selection objective does not equal arithmetic mean fold MSE")
    spans = trials[[f"s{i}" for i in (1, 2, 3, 4)]].to_numpy(dtype=float)
    if (
        not np.equal(spans, np.floor(spans)).all()
        or (spans[:, 0] < 1).any()
        or (np.diff(spans, axis=1) <= 0).any()
    ):
        raise ValueError("Need four strictly increasing integer spans")
    best = trials.iloc[int(np.argmin(trials.mean_cv_mse.to_numpy()))]
    return {
        "spans": [int(best[f"s{i}"]) for i in (1, 2, 3, 4)],
        "hyperparameters": json.loads(best.get("hyperparameters_json", "{}")),
        "selection": {
            "trial": int(best.trial),
            "mean_cv_mse_c2": float(best.mean_cv_mse),
            "fold_mse_c2": [float(best[f"fold{i}_mse"]) for i in (1, 2, 3)],
        },
    }


def linear_configuration(
    reference: dict, summary: pd.DataFrame, trials: pd.DataFrame
) -> dict:
    """Apply the original converged Stage 9K selection and C*N adjustment.

    Parameters
    ----------
    reference : dict
        Fresh larger-search RBF-SVR winner, with spans and epsilon.
    summary : pandas.DataFrame
        Combined 9J/9K four-C grouped means and warning counts.
    trials : pandas.DataFrame
        Stage 9K six records with actual fitting-row counts per fold.

    Returns
    -------
    dict
        Direct LinearSVR parameters, inherited spans and adjustment evidence.

    Raises
    ------
    ValueError
        If selection or fitting-row evidence is incomplete or nonconverged.

    Notes
    -----
    Full-data C is an extrapolation, not a separately validated full-data optimum.
    """
    if len(summary) != 4 or len(trials) != 6 or set(trials.Fold) != {1, 2, 3}:
        raise ValueError("Require four combined C values and all six Stage 9K trials")
    eligible = summary.loc[summary.Warnings.eq(0)].sort_values("Mean_MSE_C2")
    if eligible.empty or not np.isfinite(eligible[["C", "Mean_MSE_C2"]]).all().all():
        raise ValueError("No converged finite Stage 9K candidate")
    best = eligible.iloc[0]
    if (trials.groupby("Fold").Training_Rows.nunique() != 1).any():
        raise ValueError("Inconsistent reference fitting counts")
    n_reference = float(trials.drop_duplicates("Fold").Training_Rows.mean())
    if n_reference != (36 + 37 + 37) * 1280 / 3:
        raise ValueError(
            "Stage 9K must retain the original 1280/profile fitting budget"
        )
    c_reference = float(best.C)
    return {
        "spans": reference["spans"].copy(),
        "hyperparameters": {
            "C": c_reference * n_reference / 1055198,
            "epsilon": float(reference["hyperparameters"]["epsilon"]),
            "loss": "epsilon_insensitive",
            "dual": True,
            "tol": 1e-4,
            "max_iter": 10000,
        },
        "selection": {
            "mean_cv_mse_c2": float(best.Mean_MSE_C2),
            "source": "current-run Stage 9J/9K converged minimum",
        },
        "regularisation_adjustment": {
            "reference_C": c_reference,
            "mean_cv_training_rows": n_reference,
            "full_training_rows": 1055198,
            "rule": "C_full = C_reference * N_reference / N_full",
            "full_data_optimum_claimed": False,
        },
    }


def export_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    run_id: str,
    started_utc: str,
    dataset_hash: str,
    profile_cv: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    mode: str = "full",
) -> dict:
    """Export all six fresh selections with verified current-run evidence.

    Parameters
    ----------
    path : str or pathlib.Path
        Authoritative JSON destination; parent folders are created automatically.
    project_root : str or pathlib.Path
        Project root for code provenance hashing.
    run_id : str
        New UUID generated at the start of Notebook 02.
    started_utc : str
        ISO-8601 start timestamp of this execution.
    dataset_hash : str
        SHA-256 of the validated original CSV bytes.
    profile_cv : pandas.DataFrame
        Exactly the 55 development IDs and their fixed grouped folds.
    tables : dict[str, pandas.DataFrame]
        Fresh pilot, expanded, full OLS, five pilot/scaled, nested SVR,
        Stage 9H/I/J/K evidence. Required keys are documented in the guide.
    mode : str, default 'full'
        Only complete full-production runs may export an authoritative manifest.

    Returns
    -------
    dict
        Written and sealed machine-readable selection record.

    Raises
    ------
    ValueError
        If evidence is incomplete, nonfinite, inconsistent or not a full run.
    OSError
        If evidence or manifest files cannot be saved.
    """
    if mode != "full" or dataset_hash != EXPECTED_SHA256:
        raise ValueError(
            "Only full original-data development runs may publish a selection manifest"
        )
    UUID(run_id)
    _check_evidence(tables)
    if (
        sorted(profile_cv.profile_id.tolist()) != LOCKED_DEVELOPMENT_IDS
        or profile_cv.profile_id.duplicated().any()
    ):
        raise ValueError(
            "Manifest requires exactly the established development profiles"
        )
    if sorted(profile_cv.groupby("cv_fold").size().tolist()) != [18, 18, 19]:
        raise ValueError("Manifest grouped folds do not match the 19/18/18 protocol")
    required = {
        "ols_pilot": 8,
        "ols_expanded": 20,
        "ols_full": 20,
        "svr_nested": 36,
        "stage9h": 6,
        "stage9i": 15,
        "stage9j": 6,
        "stage9k": 6,
        "stage9k_summary": 4,
    }
    for name in ["SVR", "k-NN", "Random Forest", "Extra Trees", "MLP"]:
        required[f"pilot_{name}"] = 8
        required[f"scaled_{name}"] = 12
    if set(tables) != set(required):
        raise ValueError(f"Evidence table keys differ: expected {sorted(required)}")
    for name, count in required.items():
        if len(tables[name]) != count:
            raise ValueError(f"{name} needs {count} records, got {len(tables[name])}")
    select_winner(tables["ols_pilot"], 8)
    expanded = tables["ols_expanded"].copy()
    expanded["trial"] = np.arange(1, 21)
    select_winner(expanded, 20)
    models = {"OLS": select_winner(tables["ols_full"], 20)}
    references = {}
    for name in ["SVR", "k-NN", "Random Forest", "Extra Trees", "MLP"]:
        select_winner(tables[f"pilot_{name}"], 8)
        winner = select_winner(tables[f"scaled_{name}"], 12)
        winner["selection"]["source"] = "current-run 12-evaluation larger GP search"
        winner["training_samples_per_profile"] = protocol()["scaled_models"][
            "train_per_profile"
        ][name]
        winner["validation_samples_per_profile"] = 640
        (references if name == "SVR" else models)[name] = winner
    models["OLS"]["selection"][
        "source"
    ] = "current-run 20-evaluation full-development QR GP search"
    models["LinearSVR"] = linear_configuration(
        references["SVR"], tables["stage9k_summary"], tables["stage9k"]
    )
    path = Path(path).resolve()
    evidence_dir = path.parent / "runs" / run_id / "handoff_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for index, (name, table) in enumerate(tables.items()):
        target = (
            evidence_dir / f"{index:02d}_{name.replace(' ', '_').replace('-', '')}.csv"
        )
        table.to_csv(target, index=False)
        evidence[name] = {
            "path": target.relative_to(path.parent).as_posix(),
            "sha256": file_sha256(target),
            "rows": len(table),
        }
    payload = _native(
        {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "id": run_id,
                "mode": "full",
                "status": "completed",
                "scope": "complete_original_protocol",
                "started_utc": started_utc,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
            "dataset": {
                "sha256": dataset_hash,
                "rows": 1330816,
                "development_rows": 1055198,
                "heldout_rows": 275618,
                "development_profile_ids": LOCKED_DEVELOPMENT_IDS,
                "heldout_profile_ids": LOCKED_TEST_IDS,
            },
            "validation": {
                "method": "profile-level qcut(pm_max,3) + shuffled StratifiedKFold(3,seed=42)",
                "fold_by_profile": {
                    str(int(r.profile_id)): int(r.cv_fold)
                    for r in profile_cv.itertuples()
                },
                "selection_uses_heldout": False,
                "previous_heldout_inspection_disclosed": True,
            },
            "models": models,
            "svr_reference": references["SVR"],
            "search_protocol": protocol(),
            "preprocessing": FEATURE_CONTRACT,
            "training": {
                "development_rows": 1055198,
                "random_seed": 42,
                "MLP": {
                    "max_iter": 400,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                    "n_iter_no_change": 15,
                },
                "QR_OLS": {
                    "chunk_size": 4096,
                    "lapack_driver": "gelsd",
                    "rank_is_diagnostic": True,
                },
            },
            "software": software_versions(),
            "methodology_sha256": methodology_digest(project_root),
            "evidence": evidence,
            "provenance": {
                "objectives": "provenance/objective_extraction.json",
                "comparison": "docs/METHODOLOGY_PRESERVATION.md",
                "historical_values_used_as_fresh_outputs": False,
            },
        }
    )
    payload["manifest_sha256"] = _seal(payload)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)
    status_path = path.with_name("development_run_status.json")
    temporary_status = status_path.with_suffix(".json.partial")
    temporary_status.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "manifest_sha256": payload["manifest_sha256"],
            },
            indent=2,
        )
    )
    temporary_status.replace(status_path)
    return payload


def load_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    dataset_hash: str,
    profile_cv: pd.DataFrame | None = None,
) -> dict:
    """Load a complete production handoff; reject every fallback condition.

    Parameters
    ----------
    path : str or pathlib.Path
        Manifest exported by Notebook 02; no fallback path is tried.
    project_root : str or pathlib.Path
        Current project for numerical code compatibility checks.
    dataset_hash : str
        Independently computed original CSV SHA-256.
    profile_cv : pandas.DataFrame or None
        Optional independent re-derivation of development fold assignments.

    Returns
    -------
    dict
        Validated manifest. ``models`` is the sole final model configuration.

    Raises
    ------
    FileNotFoundError
        If development has not supplied the manifest or evidence files.
    ValueError
        On incomplete, smoke, incompatible, altered or unsupported contents.

    Notes
    -----
    This verifies evidence integrity and consistency, not authorship or the
    truthfulness of an externally fabricated manifest. Never load untrusted
    pickle/joblib artifacts. This loader reads JSON and CSV only.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Development manifest missing: {path}. Run Notebook 02 completely before Notebook 01."
        )
    try:
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported development manifest schema")
        if payload.get("manifest_sha256") != _seal(payload):
            raise ValueError("Manifest checksum mismatch")
        run = payload["run"]
        UUID(run["id"])
        status_path = path.with_name("development_run_status.json")
        if not status_path.is_file():
            raise FileNotFoundError(
                "Development run-status evidence missing; transfer the complete handoff archive from Notebook 02."
            )
        status = json.loads(status_path.read_text())
        if (
            status.get("run_id"),
            status.get("status"),
            status.get("manifest_sha256"),
        ) != (run["id"], "completed", payload["manifest_sha256"]):
            raise ValueError(
                "Development run is incomplete or superseded; do not reuse an earlier manifest"
            )
        if (run["mode"], run["status"], run["scope"]) != (
            "full",
            "completed",
            "complete_original_protocol",
        ):
            raise ValueError(
                "Final evaluation requires a completed original-budget production run"
            )
        data = payload["dataset"]
        if dataset_hash != EXPECTED_SHA256 or data["sha256"] != dataset_hash:
            raise ValueError("Dataset hash differs from current development selection")
        if (
            data["development_profile_ids"] != LOCKED_DEVELOPMENT_IDS
            or data["heldout_profile_ids"] != LOCKED_TEST_IDS
        ):
            raise ValueError("Development/held-out profile assignments changed")
        if [data[k] for k in ["rows", "development_rows", "heldout_rows"]] != [
            1330816,
            1055198,
            275618,
        ]:
            raise ValueError("Dataset row counts changed")
        if (
            payload["search_protocol"] != protocol()
            or payload["preprocessing"] != FEATURE_CONTRACT
        ):
            raise ValueError(
                "Search budgets, dimensions or preprocessing do not match the recovered protocol"
            )
        if payload["software"] != software_versions():
            raise ValueError(
                "Numerical software versions differ; use the development environment or rerun Notebook 02"
            )
        if payload["methodology_sha256"] != methodology_digest(project_root):
            raise ValueError(
                "Production source differs from the development run; regenerate selection"
            )
        if payload["validation"]["selection_uses_heldout"] is not False:
            raise ValueError("Held-out results cannot be used for selection")
        if profile_cv is not None:
            expected = {
                str(int(r.profile_id)): int(r.cv_fold) for r in profile_cv.itertuples()
            }
            if expected != payload["validation"]["fold_by_profile"]:
                raise ValueError(
                    "Grouped-CV assignments differ from development evidence"
                )
        if set(payload["models"]) != set(TRAIN_ORDER):
            raise ValueError("All six selected final model configurations are required")
        tables = {}
        for name, record in payload["evidence"].items():
            target = (path.parent / record["path"]).resolve()
            if not target.is_relative_to(path.parent):
                raise ValueError("Evidence path escapes the manifest directory")
            if file_sha256(target) != record["sha256"]:
                raise ValueError(f"Development evidence changed: {name}")
            table = pd.read_csv(target, float_precision="round_trip")
            if len(table) != record["rows"]:
                raise ValueError(f"Development evidence row count changed: {name}")
            tables[name] = table
        _check_evidence(tables)
        expected_keys = {
            "ols_pilot",
            "ols_expanded",
            "ols_full",
            "svr_nested",
            "stage9h",
            "stage9i",
            "stage9j",
            "stage9k",
            "stage9k_summary",
        }
        expected_keys |= {
            f"{prefix}_{name}"
            for prefix in ["pilot", "scaled"]
            for name in ["SVR", "k-NN", "Random Forest", "Extra Trees", "MLP"]
        }
        if set(tables) != expected_keys:
            raise ValueError("Required search and numerical evidence is incomplete")
        checks = {"OLS": select_winner(tables["ols_full"], 20)}
        for name in ["k-NN", "Random Forest", "Extra Trees", "MLP"]:
            checks[name] = select_winner(tables[f"scaled_{name}"], 12)
        ref = select_winner(tables["scaled_SVR"], 12)
        for key in ["spans", "hyperparameters", "selection"]:
            # Source label is descriptive; numeric selection is checked separately.
            a, b = payload["svr_reference"][key], ref[key]
            if key == "selection":
                a = {k: v for k, v in a.items() if k != "source"}
            if a != b:
                raise ValueError(
                    "RBF-SVR reference does not match fresh search evidence"
                )
        checks["LinearSVR"] = linear_configuration(
            ref, tables["stage9k_summary"], tables["stage9k"]
        )
        for name, expected in checks.items():
            actual = payload["models"][name]
            if any(
                actual[key] != expected[key] for key in ["spans", "hyperparameters"]
            ):
                raise ValueError(
                    f"{name} differs from its current-run selection evidence"
                )
            if (
                actual["selection"]["mean_cv_mse_c2"]
                != expected["selection"]["mean_cv_mse_c2"]
            ):
                raise ValueError(f"{name} selection score differs from its evidence")
            for key, value in expected["selection"].items():
                if key != "source" and actual["selection"].get(key) != value:
                    raise ValueError(
                        f"{name} selection details differ from its evidence"
                    )
            if (
                name == "LinearSVR"
                and actual.get("regularisation_adjustment")
                != expected["regularisation_adjustment"]
            ):
                raise ValueError(
                    "LinearSVR regularisation adjustment differs from its evidence"
                )
        from .models import build_pipeline

        for name in TRAIN_ORDER:
            if name != "OLS":
                build_pipeline(name, payload["models"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Incomplete or invalid development manifest: {exc}") from exc
    return payload


def configuration_agreement(manifest: dict, configurations: dict) -> pd.DataFrame:
    """Verify and display every final model against the imported manifest.

    Parameters
    ----------
    manifest : dict
        Validated output of ``load_manifest``.
    configurations : dict
        Actual mapping that final factories and feature builders will use.

    Returns
    -------
    pandas.DataFrame
        Six rows with spans, parameter JSON, source and agreement flag.

    Raises
    ------
    ValueError
        If any model configuration differs before fitting begins.
    """
    if configurations != manifest["models"]:
        raise ValueError(
            "Final training configurations diverge from imported development manifest"
        )
    return pd.DataFrame(
        [
            {
                "Model": name,
                "Spans_samples": str(configurations[name]["spans"]),
                "Hyperparameters": json.dumps(
                    configurations[name]["hyperparameters"], sort_keys=True
                ),
                "Selection_source": configurations[name]["selection"]["source"],
                "Agrees": True,
            }
            for name in TRAIN_ORDER
        ]
    )


def write_handoff_archive(
    path: str | Path, destination: str | Path | None = None
) -> Path:
    """Bundle the completed selection and required evidence for a new runtime.

    Parameters
    ----------
    path : str or pathlib.Path
        Manifest written by the successful current development run.
    destination : str, pathlib.Path or None
        Output ZIP. None uses ``development_handoff.zip`` beside the manifest.

    Returns
    -------
    pathlib.Path
        Portable archive with manifest, run-status marker and evidence CSVs.
        Extract it into the final runtime's ``results/full/development/``.

    Raises
    ------
    FileNotFoundError, OSError
        If any required evidence file is missing or an archive cannot be written.
    ValueError
        If the manifest, completion marker or evidence hashes disagree.

    Notes
    -----
    This copies existing evidence. It never trains, optimises, loads a model,
    changes a parameter or includes the dataset and large trained estimators.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (
        manifest.get("manifest_sha256") != _seal(manifest)
        or manifest["run"]["status"] != "completed"
    ):
        raise ValueError("Cannot archive an incomplete or changed manifest")
    marker = path.with_name("development_run_status.json")
    status = json.loads(marker.read_text())
    if (status.get("run_id"), status.get("status"), status.get("manifest_sha256")) != (
        manifest["run"]["id"],
        "completed",
        manifest["manifest_sha256"],
    ):
        raise ValueError("Cannot archive an incomplete or superseded development run")
    files = [path, marker]
    for record in manifest["evidence"].values():
        target = (path.parent / record["path"]).resolve()
        if (
            not target.is_relative_to(path.parent)
            or file_sha256(target) != record["sha256"]
        ):
            raise ValueError("Cannot archive changed or out-of-directory evidence")
        files.append(target)
    destination = (
        Path(destination).resolve()
        if destination is not None
        else path.with_name("development_handoff.zip")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for target in files:
            archive.write(target, target.relative_to(path.parent))
    return destination
