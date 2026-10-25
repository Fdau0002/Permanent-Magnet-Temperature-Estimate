from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from pmsm_project.data import (
    file_sha256,
    verify_csv_bytes,
    validate_dataset,
    resolve_csv,
)
from pmsm_project.config import EXPECTED_SHA256


def test_chunked_hash_independent(tmp_path):
    p = tmp_path / "bytes.bin"
    payload = bytes(range(256)) * 21000
    p.write_bytes(payload)
    assert file_sha256(p) == hashlib.sha256(payload).hexdigest()


def test_checksum_rejects_modified_bytes(tmp_path):
    p = tmp_path / "wrong.csv"
    p.write_text("pm,profile_id\n2,2\n")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_csv_bytes(p)


def test_missing_csv_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_csv_bytes(tmp_path / "absent.csv")


def test_manual_missing_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_csv(tmp_path, "missing.csv")


def test_no_download_alternative(tmp_path, monkeypatch):
    monkeypatch.delenv("PMSM_CSV_PATH", raising=False)
    with pytest.raises(FileNotFoundError):
        resolve_csv(tmp_path, allow_download=False)


def test_valid_synthetic_contract(profile):
    a = validate_dataset(profile, strict=False)
    assert a["observations"] == 360 and a["profiles"] == 1 and a["duplicates"] == 0


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "missing_column",
        "order",
        "nan",
        "inf",
        "text",
        "fractional_id",
        "duplicate",
        "interleaved",
        "index",
    ],
)
def test_invalid_data_contracts(profile, problem):
    f = profile.copy()
    if problem == "empty":
        f = f.iloc[:0]
    elif problem == "missing_column":
        f = f.drop(columns="pm")
    elif problem == "order":
        f = f[list(reversed(f.columns))]
    elif problem == "nan":
        f.loc[3, "i_q"] = np.nan
    elif problem == "inf":
        f.loc[3, "pm"] = np.inf
    elif problem == "text":
        f["pm"] = "bad"
    elif problem == "fractional_id":
        f["profile_id"] = 2.5
    elif problem == "duplicate":
        f = pd.concat([f, f.iloc[[0]]], ignore_index=True)
    elif problem == "interleaved":
        f.loc[3, "profile_id"] = 3
    elif problem == "index":
        f.index = np.arange(len(f))[::-1]
    with pytest.raises(ValueError):
        validate_dataset(f, strict=False)


def test_strict_rejects_fixture(profile):
    with pytest.raises(ValueError, match="established"):
        validate_dataset(profile)


@pytest.mark.integration
@pytest.mark.slow
def test_original_bytes_and_dataset(original_csv):
    assert verify_csv_bytes(original_csv) == EXPECTED_SHA256
    a = validate_dataset(pd.read_csv(original_csv))
    assert a == dict(
        observations=1330816,
        profiles=69,
        contiguous_blocks=69,
        duplicates=0,
        finite=True,
    )


def test_public_download_error_gives_manual_route(tmp_path, monkeypatch):
    import kagglehub

    monkeypatch.delenv("PMSM_CSV_PATH", raising=False)

    def unavailable(handle):
        raise ConnectionError("Simulated network unavailable")

    monkeypatch.setattr(kagglehub, "dataset_download", unavailable)
    with pytest.raises(
        RuntimeError, match="manually|Download measures_v2|original CSV"
    ):
        resolve_csv(tmp_path)


def test_downloaded_bytes_still_require_integrity(tmp_path, monkeypatch):
    import kagglehub

    monkeypatch.delenv("PMSM_CSV_PATH", raising=False)
    (tmp_path / "measures_v2.csv").write_text("changed source")
    download_root = tmp_path / "cache"
    download_root.mkdir()
    (download_root / "measures_v2.csv").write_text("changed source")
    monkeypatch.setattr(
        kagglehub, "dataset_download", lambda handle: str(download_root)
    )
    with pytest.raises(ValueError, match="bytes changed"):
        resolve_csv(tmp_path / "different_root")
