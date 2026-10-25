"""Original physical-temperature evaluation equations (Celsius).

Residuals are prediction minus measurement. Constant targets return NaN R²,
as in the supplied final notebook; they are not coerced to a finite score.
"""

import numpy as np


def score(y: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    """Evaluate the exact supplied final-notebook temperature metrics.

    Parameters
    ----------
    y : array-like of float, shape (n,)
        Measured permanent-magnet temperatures, deg C.
    yp : array-like of float, shape (n,)
        Row-aligned predictions, deg C.

    Returns
    -------
    dict of str to float
        MSE_C2 (deg C squared), RMSE_C, MAE_C, Bias_C, P95_AE_C and
        Max_AE_C (deg C), and dimensionless R2. Bias is prediction minus
        measurement; positive values mean average overprediction.

    Raises
    ------
    ValueError
        If arrays are empty, nonfinite, not vectors or have unequal length.

    Notes
    -----
    R2 is NaN for constant measured targets, preserving the source convention.
    Percentile interpolation uses NumPy's default linear method. Pooled scores
    weight observations equally; mean profile scores weight sessions equally.
    Neither score estimates a temperature-limit failure probability."""
    y, yp = np.asarray(y, dtype=float), np.asarray(yp, dtype=float)
    if y.ndim != 1 or yp.ndim != 1 or y.size == 0 or y.shape != yp.shape:
        raise ValueError("Require nonempty, equal-length vectors")
    if not np.isfinite(y).all() or not np.isfinite(yp).all():
        raise ValueError("Require finite measurements and predictions")
    e = np.asarray(yp, dtype=float) - np.asarray(y, dtype=float)
    mse = float(np.mean(e**2))
    den = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "MSE_C2": mse,
        "RMSE_C": float(np.sqrt(mse)),
        "MAE_C": float(np.mean(abs(e))),
        "R2": float(1 - np.sum(e**2) / den) if den > 0 else np.nan,
        "Bias_C": float(np.mean(e)),
        "P95_AE_C": float(np.percentile(abs(e), 95)),
        "Max_AE_C": float(np.max(abs(e))),
    }
