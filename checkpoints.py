"""Read row-aligned prediction evidence without accepting another model or run."""

from pathlib import Path
import numpy as np


def load_prediction_checkpoint(
    path: str | Path,
    *,
    model: str,
    profile_id: int,
    source_rows: np.ndarray,
    mode: str,
    artifact_sha256: str,
    manifest_sha256: str
) -> np.ndarray:
    """Verify prediction identity before computing any temperature metrics.

    Parameters
    ----------
    path : str or pathlib.Path
        NPZ produced by the final notebook's actual inference loop.
    model : str
        Expected estimator family.
    profile_id : int
        Expected independent recording ID.
    source_rows : array-like of int
        Ordered original CSV row indices for that recording.
    mode : str
        Expected full or smoke run label.
    artifact_sha256, manifest_sha256 : str
        Expected model-file and imported-selection SHA-256 digests.

    Returns
    -------
    numpy.ndarray
        Verified finite predicted temperatures in degrees Celsius, in source order.

    Raises
    ------
    FileNotFoundError, OSError
        If the checkpoint cannot be read.
    ValueError
        On missing fields, identity mismatch, wrong row ordering, shape or finiteness.

    Notes
    -----
    Loading uses allow_pickle=False. These checks add provenance validation;
    the original prediction values and metric equations are not changed.
    """
    expected = {
        "model": model,
        "profile_id": int(profile_id),
        "mode": mode,
        "artifact_sha256": artifact_sha256,
        "selection_manifest_sha256": manifest_sha256,
    }
    with np.load(path, allow_pickle=False) as archive:
        try:
            if any(archive[key].item() != value for key, value in expected.items()):
                raise ValueError(
                    "Prediction checkpoint belongs to a different model, profile or selection run"
                )
            rows = archive["source_row"]
            predictions = archive["prediction"].copy()
        except (KeyError, TypeError) as exc:
            raise ValueError("Prediction checkpoint has an incomplete schema") from exc
    if not np.array_equal(rows, np.asarray(source_rows, dtype=np.int64)):
        raise ValueError(
            "Prediction source-row order differs from the original measurements"
        )
    if (
        predictions.ndim != 1
        or predictions.shape != rows.shape
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("Prediction vector must be finite and match the source rows")
    return predictions
