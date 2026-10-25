"""Public dataset acquisition, exact-byte integrity and strict input contracts."""

from __future__ import annotations
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .config import (
    EXPECTED_SHA256,
    EXPECTED_COLUMNS,
    DATASET_HANDLE,
    LOCKED_DEVELOPMENT_IDS,
    LOCKED_TEST_IDS,
)


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 without loading an entire dataset into memory.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing binary file; no physical units.

    Returns
    -------
    str
        Lowercase 64-digit hexadecimal digest of unmodified bytes.

    Raises
    ------
    OSError
        If the file cannot be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_csv_bytes(path: str | Path) -> str:
    """Require the exact source CSV used by both uploaded investigations.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file with original bytes (renaming is allowed).

    Returns
    -------
    str
        Verified SHA-256 digest.

    Raises
    ------
    FileNotFoundError
        If no regular file exists.
    ValueError
        If bytes differ; no reserialization or checksum bypass is performed.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing source CSV: {p}")
    actual = file_sha256(p)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"Dataset bytes changed: expected {EXPECTED_SHA256}, got {actual}"
        )
    return actual


def resolve_csv(
    project_root: str | Path,
    local_path: str | Path | None = None,
    allow_download: bool = True,
) -> Path:
    """Resolve an explicit local CSV or download public dataset version 3.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Extracted repository directory.
    local_path : str, pathlib.Path or None
        Optional exact path. Otherwise ``PMSM_CSV_PATH`` then
        ``data/measures_v2.csv`` is used. Relative paths are project-relative.
    allow_download : bool, default True
        Allow KaggleHub's public download when no local file exists.

    Returns
    -------
    pathlib.Path
        Absolute path to hash-verified original CSV bytes.

    Raises
    ------
    FileNotFoundError
        If a requested path is absent or downloading is disabled.
    RuntimeError
        If public retrieval fails; the message gives the manual alternative.
    ValueError
        If the resolved data version is different.

    Notes
    -----
    Kaggle may impose network, authentication or availability restrictions.
    No personal Drive, Kaggle credential file, or historical model is needed.
    The dataset remains in the caller's cache; it is not bundled in the ZIP.
    """
    root = Path(project_root).resolve()
    supplied = local_path or os.environ.get("PMSM_CSV_PATH", "").strip()
    if supplied:
        candidate = Path(supplied).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        verify_csv_bytes(candidate)
        return candidate.resolve()
    candidate = root / "data" / "measures_v2.csv"
    if not candidate.is_file():
        if not allow_download:
            raise FileNotFoundError(
                "Place original measures_v2.csv in data/ or set PMSM_CSV_PATH"
            )
        try:
            import kagglehub

            candidate = (
                Path(kagglehub.dataset_download(DATASET_HANDLE)) / "measures_v2.csv"
            )
        except Exception as exc:
            raise RuntimeError(
                "Public Kaggle retrieval failed. Download measures_v2.csv "
                "from wkirgsn/electric-motor-temperature (version 3), "
                "place it in data/, or set PMSM_CSV_PATH to its exact path."
            ) from exc
    verify_csv_bytes(candidate)
    return candidate.resolve()


def validate_dataset(frame: pd.DataFrame, strict: bool = True) -> dict:
    """Validate schema, numeric values and one contiguous block per session.

    Parameters
    ----------
    frame : pandas.DataFrame
        Raw 13-column PMSM measurements in original chronological row order.
        Temperatures are deg C, speed rpm, dq voltages V, dq currents A,
        torque N m, and profile IDs dimensionless.
    strict : bool, default True
        Require the exact 1,330,816-row, 69-profile study dataset as well as
        general schema checks. False supports small synthetic unit fixtures.

    Returns
    -------
    dict
        Row/profile/block counts and duplicate count for an audit.

    Raises
    ------
    ValueError
        On schema/order mismatch, empty input, invalid numeric entries,
        nonintegral IDs, duplicate rows or interleaved profile blocks.

    Notes
    -----
    Structural checks cannot validate sensor calibration or operating safety.
    Invalid measurements are rejected; none are imputed, clipped or dropped.
    """
    if list(frame.columns) != EXPECTED_COLUMNS or frame.empty:
        raise ValueError(
            "Require a nonempty original 13-column schema in recorded order"
        )
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Require unique increasing original source-row indices")
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("All source measurements must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(
            "Missing or nonfinite measurement; no implicit cleaning is permitted"
        )
    ids = frame.profile_id.to_numpy(dtype=float)
    if not np.equal(ids, np.floor(ids)).all():
        raise ValueError("Profile IDs must be integers")
    blocks = int(frame.profile_id.ne(frame.profile_id.shift()).sum())
    profiles = int(frame.profile_id.nunique())
    duplicates = int(frame.duplicated().sum())
    if blocks != profiles or duplicates:
        raise ValueError("Interleaved profiles or duplicate complete observations")
    if strict and (
        len(frame) != 1_330_816
        or profiles != 69
        or set(ids) != set(LOCKED_DEVELOPMENT_IDS + LOCKED_TEST_IDS)
    ):
        raise ValueError(
            "Not the established 1,330,816-observation / 69-profile dataset"
        )
    return dict(
        observations=len(frame),
        profiles=profiles,
        contiguous_blocks=blocks,
        duplicates=duplicates,
        finite=True,
    )
