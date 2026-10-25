"""Paper-inspired predictors and profile-local finite-history exponential moments.

All spans are sample counts at 2 Hz. Neither target temperature nor profile ID
enters the 108 predictors. The vectorised sampler deliberately retains its
original span-major insertion order. Stage 5 retains that order consistently.
Stage 6/Stage 9 explicitly select variable-major canonical order; the final
feature_frame supplies it directly. Model schemas are not mixed. See Kirchgässner et al. (2021).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .config import BASE_COLS


def make_paper_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Construct the 12 instantaneous paper-inspired PMSM predictors.

    Parameters
    ----------
    data : pandas.DataFrame
        Original PMSM measurements, with ambient, coolant, u_d, u_q,
        i_d, i_q and motor_speed in the dataset's original units.

    Returns
    -------
    pandas.DataFrame
        Twelve predictors, in the original row order and index; no
        target, session identifier, or downstream fitted parameters.

    Raises
    ------
    ValueError
        If one of the seven measured input columns is missing.

    Notes
    -----
    Ambient/coolant temperatures use deg C, dq voltages V, dq currents A,
    and speed rpm. Derived magnitudes, apparent-power proxy and speed
    interactions inherit the corresponding compound units. Sensor
    calibration and physically safe operating limits are not inferred.
    """
    required = ("ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed")
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing measured inputs: {missing}")

    X = pd.DataFrame(index=data.index)
    X["theta_a"] = data["ambient"]
    X["theta_c"] = data["coolant"]
    X["u_d"] = data["u_d"]
    X["u_q"] = data["u_q"]
    X["i_d"] = data["i_d"]
    X["i_q"] = data["i_q"]
    X["n_mech"] = data["motor_speed"]

    X["u_s"] = np.hypot(X["u_d"], X["u_q"])
    X["i_s"] = np.hypot(X["i_d"], X["i_q"])
    X["S_el"] = 1.5 * X["u_s"] * X["i_s"]
    omega = (2 * np.pi / 60) * X["n_mech"]
    X["i_s_omega"] = X["i_s"] * omega
    X["S_el_omega"] = X["S_el"] * omega

    expected = (
        "theta_a",
        "theta_c",
        "u_d",
        "u_q",
        "i_d",
        "i_q",
        "n_mech",
        "u_s",
        "i_s",
        "S_el",
        "i_s_omega",
        "S_el_omega",
    )
    assert tuple(X.columns) == expected
    assert "pm" not in X.columns and "profile_id" not in X.columns
    return X


def make_temporal_features(
    X: pd.DataFrame, profile_ids: pd.Series, spans=(10, 50, 200, 1000)
) -> pd.DataFrame:
    """Add causal EWMA and EWMS features, restarting at each profile.

    Parameters
    ----------
    X : pandas.DataFrame
        Numeric predictors in chronological (original dataset row) order.
    profile_ids : pandas.Series
        Recording-session IDs aligned with X.index. Each session must
        appear in a single chronologically ordered block.
    spans : sequence of positive integers
        EWMA smoothing spans in sample counts (not seconds). These are
        illustrative until cross-validated in the modelling stage.

    Returns
    -------
    pandas.DataFrame
        Original columns followed by EWMA/EWMS pairs for each input
        and span; 12 + 2*12*4 = 108 columns for the default values.

    Raises
    ------
    TypeError
        If the predictors/IDs are not a DataFrame and aligned Series.
    ValueError
        If input is empty, missing/nonfinite, misaligned, nonchronological,
        interleaved across sessions, or spans are not distinct positive integers.

    Notes
    -----
    adjust=True implements a finite-history, normalized EW mean.
    std(bias=True) produces its population weighted standard deviation.
    These are feature transformations, not train-fitted scalers.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(profile_ids, pd.Series):
        raise TypeError("X must be a DataFrame and profile_ids must be a Series")
    if not X.index.equals(profile_ids.index):
        raise ValueError("X and profile_ids must have exactly aligned indices")
    if not X.index.is_unique or X.empty or profile_ids.isna().any():
        raise ValueError(
            "Require unique row index, nonempty data, and valid profile IDs"
        )
    if not X.index.is_monotonic_increasing:
        raise ValueError("Preserve increasing original source-row order")
    if int(profile_ids.ne(profile_ids.shift()).sum()) != profile_ids.nunique():
        raise ValueError("Each profile must occupy one contiguous block")
    spans = tuple(spans)
    if (
        not spans
        or len(set(spans)) != len(spans)
        or any(not isinstance(s, (int, np.integer)) or s < 1 for s in spans)
    ):
        raise ValueError("Spans must be distinct positive integers")
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Inputs must have finite nonmissing values")

    # These features are calculated within a profile, in existing row order.
    # Note that the user must preserve chronological order before calling.
    engineered = {}
    for column in X.columns:
        grouped = X[column].groupby(profile_ids, sort=False)
        for s in spans:
            ewma_name = f"{column}_ewma_s{s}"
            ewms_name = f"{column}_ewms_s{s}"
            engineered[ewma_name] = grouped.transform(
                lambda v: v.ewm(span=s, adjust=True).mean()
            )
            engineered[ewms_name] = grouped.transform(
                lambda v: v.ewm(span=s, adjust=True).std(bias=True)
            )

    engineered_frame = pd.DataFrame(engineered, index=X.index)
    return pd.concat([X, engineered_frame], axis=1)


def vectorised_temporal_sample(
    X_original: pd.DataFrame, positions: np.ndarray, spans: tuple | list
) -> pd.DataFrame:
    """Sample profile-local exponential moments after complete-history filtering.

    Parameters
    ----------
    X_original : pandas.DataFrame
        Twelve instantaneous predictors for one recording in original row order.
        Mixed physical units are inherited from ``make_paper_inputs``.
    positions : array-like of int
        Zero-based within-profile positions, selected after filtering.
    spans : sequence of int
        Four smoothing spans in samples at 2 Hz; adjust=True and bias=True.

    Returns
    -------
    pandas.DataFrame
        Selected rows, indexed by original source row. Columns are base inputs
        followed by span-major EWMA/EWMS pairs. Use ``stage6_feature_names``
        to select the canonical variable-major model order.

    Raises
    ------
    IndexError
        If a position is outside the profile.
    ValueError
        If pandas cannot interpret a smoothing span.

    Notes
    -----
    No fitted normalisation is performed. The caller must supply one complete
    chronological profile; passing disjoint or shuffled histories changes the
    meaning of the features. The original sampler is retained verbatim."""
    sampled_columns = {
        col: X_original[col].iloc[positions].to_numpy() for col in X_original.columns
    }
    for s in spans:
        ewm = X_original.ewm(span=int(s), adjust=True)
        historical_mean = ewm.mean().iloc[positions]
        historical_std = ewm.std(bias=True).iloc[positions]
        for col in X_original.columns:
            sampled_columns[f"{col}_ewma_s{s}"] = historical_mean[col].to_numpy()
            sampled_columns[f"{col}_ewms_s{s}"] = historical_std[col].to_numpy()
    return pd.DataFrame(sampled_columns, index=X_original.index[positions])


def stage6_feature_names(columns: list | tuple, spans: list | tuple) -> list[str]:
    """Construct the development canonical model feature order.

    Parameters
    ----------
    columns : sequence of str
        Ordered instantaneous input names (normally twelve).
    spans : sequence of int
        Ordered spans in samples; normally four.

    Returns
    -------
    list of str
        Raw names then EWMA/EWMS pairs per variable and span, normally 108.

    Notes
    -----
    Column ordering is part of a fitted model's interface. This function does
    not sort spans or infer units. It deliberately preserves the caller's order."""
    return list(columns) + [
        name
        for col in columns
        for span in spans
        for name in (f"{col}_ewma_s{span}", f"{col}_ewms_s{span}")
    ]


def feature_names(spans: list | tuple) -> list[str]:
    """Return the final estimator's canonical 108-column interface.

    Parameters
    ----------
    spans : sequence of int
        Four frozen smoothing spans in samples.

    Returns
    -------
    list of str
        Twelve base names plus paired temporal moments in model order.

    Notes
    -----
    Feature moments retain the units of each base predictor. Names must match
    both persisted model metadata and inference matrices exactly."""
    return BASE_COLS + [
        key
        for col in BASE_COLS
        for s in spans
        for key in (f"{col}_ewma_s{s}", f"{col}_ewms_s{s}")
    ]


def feature_frame(profile: pd.DataFrame, spans: list | tuple) -> pd.DataFrame:
    """Compute the original final-notebook feature matrix for one session.

    Parameters
    ----------
    profile : pandas.DataFrame
        Exactly one recording, in increasing original source-row order.
        Seven sensor columns plus ``profile_id`` are required.
    spans : sequence of int
        Four frozen exponential smoothing spans (samples).

    Returns
    -------
    pandas.DataFrame
        Finite float-valued matrix with 108 columns in canonical order.

    Raises
    ------
    AssertionError
        If profiles mix, ordering is invalid, shape differs or values are nonfinite.
    KeyError
        If required measured columns are absent.

    Notes
    -----
    Every EWMA uses adjust=True and every EWMS uses bias=True. The initial
    mean is the first observation and the initial population standard deviation
    is zero. Filtering uses current and previous inputs, never future values.
    Assertions retain the supplied notebook's valid-input behaviour. Do not run
    Python with -O because experimental assertions are part of its audit."""
    assert (
        profile.profile_id.nunique() == 1
    ), "Never mix recording sessions before temporal feature construction."
    assert (
        profile.index.is_monotonic_increasing
    ), "Original chronological ordering is mandatory."
    base = make_paper_inputs(profile)
    out = {}
    for col in BASE_COLS:
        for s in spans:
            ew = base[col].ewm(span=int(s), adjust=True)
            out[f"{col}_ewma_s{s}"] = ew.mean().to_numpy()
            out[f"{col}_ewms_s{s}"] = ew.std(bias=True).to_numpy()
    extra = pd.DataFrame(out, index=base.index)
    result = pd.concat([base, extra], axis=1)
    assert list(result.columns) == feature_names(spans)
    assert result.shape == (len(profile), 108) and np.isfinite(result.to_numpy()).all()
    return result
