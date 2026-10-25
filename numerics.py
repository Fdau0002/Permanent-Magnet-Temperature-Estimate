"""Profile-wise augmented QR compression and unregularised least squares.

For A=[1,X,y]=QR, orthogonality gives ||A w||²=||R w||². Local R
factors can therefore be stacked without forming the normal equations.
Development and final compressors retain their distinct original feature
construction implementations to preserve the original numerical/timing paths.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.linalg import qr, lstsq
from .features import (
    make_paper_inputs,
    make_temporal_features,
    vectorised_temporal_sample,
    stage6_feature_names,
    feature_frame,
    feature_names,
)


def compress_profile_ols(profile: pd.DataFrame, spans, chunk_size=4096):
    """Compress a single complete chronological profile's [1, X, y] using QR.

    Parameters
    ----------
    profile : pandas.DataFrame
        Exactly one recording session, unshuffled, in original sample order.
    spans : sequence[int]
        Exactly four positive, increasing temporal-smoothing spans (in samples).
    chunk_size : int
        Maximum raw observations in each temporary QR block.

    Returns
    -------
    dict
        n, feature_mean, feature_variance, R (augmented QR factor).
    """
    spans = tuple(map(int, spans))
    if len(spans) != 4 or any(a >= b for a, b in zip(spans, spans[1:])):
        raise ValueError("Need four increasing spans")
    if profile.profile_id.nunique() != 1 or not profile.index.is_monotonic_increasing:
        raise ValueError("Require one chronologically ordered recording profile")
    if chunk_size < 2:
        raise ValueError("chunk_size must be >=2")

    raw = make_paper_inputs(profile)
    temporal = make_temporal_features(raw, profile["profile_id"], spans)
    X = temporal.to_numpy(dtype=np.float64)
    y = profile["pm"].to_numpy(dtype=np.float64)
    n, p = X.shape
    assert p == 108 and n == len(y)
    assert np.isfinite(X).all() and np.isfinite(y).all()

    mu = X.mean(axis=0)
    variance = X.var(axis=0)  # Population variance, same convention as StandardScaler
    R_profile = None
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = np.empty((stop - start, p + 2), dtype=np.float64)
        block[:, 0] = 1.0
        block[:, 1:-1] = X[start:stop]
        block[:, -1] = y[start:stop]
        R_block = qr(block, mode="r", check_finite=False)[0][: p + 2, :]
        if R_profile is None:
            R_profile = R_block
        else:
            R_profile = qr(
                np.vstack((R_profile, R_block)), mode="r", check_finite=False
            )[0][: p + 2, :]
    return {"n": n, "mean": mu, "var": variance, "R": R_profile}


def qr_compress_profile(
    profile: pd.DataFrame, spans: list | tuple, chunk: int = 4096
) -> dict:
    """Compress the original final-notebook augmented profile design by QR.

    Parameters
    ----------
    profile : pandas.DataFrame
        One chronological recording with target ``pm`` in deg C.
    spans : sequence of int
        Four model-specific smoothing spans in samples.
    chunk : int, default 4096
        Raw observations per QR block; the full profile feature matrix is
        still constructed, so this is not a streaming feature implementation.

    Returns
    -------
    dict
        ``n``, population feature ``mean`` and ``var``, and augmented ``R``
        for [1, X, y]. R factors preserve squared residual norms, not QR signs.

    Raises
    ------
    AssertionError
        If feature construction violates the single-session contract.
    ValueError
        If the chunk cannot be used as a positive integer step.

    Notes
    -----
    Float64 scipy.linalg.qr uses mode='r'. Dense R workspaces may be larger
    than the retained triangular factor. Rank deficiency is diagnosed by the
    subsequent SVD least-squares solve, not corrected here."""
    f = feature_frame(profile, spans)
    X = f.to_numpy(dtype=np.float64)
    y = profile.pm.to_numpy(dtype=np.float64)
    p = X.shape[1]
    R = None
    for start in range(0, len(profile), chunk):
        stop = min(start + chunk, len(profile))
        A = np.empty((stop - start, p + 2), dtype=np.float64)
        A[:, 0] = 1.0
        A[:, 1:-1] = X[start:stop]
        A[:, -1] = y[start:stop]
        rb = qr(A, mode="r", check_finite=False)[0][: p + 2, :]
        R = (
            rb
            if R is None
            else qr(np.vstack((R, rb)), mode="r", check_finite=False)[0][: p + 2, :]
        )
    return {"n": len(profile), "mean": X.mean(axis=0), "var": X.var(axis=0), "R": R}


def fit_qr_ols(data: pd.DataFrame, spans: list | tuple) -> dict:
    """Fit the supplied profile-wise QR ordinary least-squares estimator.

    Parameters
    ----------
    data : pandas.DataFrame
        Complete training recordings; target ``pm`` is in deg C. No held-out
        profiles may be supplied. Original row indices preserve time order.
    spans : sequence of int
        Four fixed exponential-memory spans in samples.

    Returns
    -------
    dict
        Intercept (deg C), coefficients (deg C per predictor unit), feature
        names, spans, training row count, numerical rank and fit_seconds.
        The timer includes feature construction and QR, unlike sklearn fit timers.

    Raises
    ------
    AssertionError
        If fitted coefficients are nonfinite or profile feature checks fail.
    ValueError
        If no profile factors can be stacked or a numerical solve is invalid.

    Notes
    -----
    Profile population moments are combined with the between-profile mean
    correction. Scaling is applied to the QR factors using their intercept
    column; scipy.linalg.lstsq(lapack_driver='gelsd') solves the scaled problem.
    An explicit normal-equation inverse is never formed. Rank is sensitive to
    numerical precision and feature collinearity."""
    stats = []
    t0 = time.perf_counter()
    for pid, profile in data.groupby("profile_id", sort=False):
        stats.append(qr_compress_profile(profile, spans))
    count = sum(item["n"] for item in stats)
    mean = sum(item["n"] * item["mean"] for item in stats) / count
    var = (
        sum(item["n"] * (item["var"] + (item["mean"] - mean) ** 2) for item in stats)
        / count
    )
    scale = np.sqrt(np.maximum(var, 0))
    scale[scale == 0] = 1
    R = np.vstack([item["R"] for item in stats])
    Rscaled = R.copy()
    Rscaled[:, 1:-1] = (R[:, 1:-1] - R[:, [0]] * mean) / scale
    w, _, rank, _ = lstsq(Rscaled[:, :-1], Rscaled[:, -1], lapack_driver="gelsd")
    coef = w[1:] / scale
    intercept = float(w[0] - np.dot(mean, coef))
    assert np.isfinite(coef).all() and np.isfinite(intercept)
    return {
        "model": "OLS",
        "coef": coef,
        "intercept": intercept,
        "spans": spans,
        "feature_names": feature_names(spans),
        "train_rows": count,
        "rank": int(rank),
        "fit_seconds": time.perf_counter() - t0,
    }
