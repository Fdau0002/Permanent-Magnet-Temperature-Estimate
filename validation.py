"""Profile-level allocation and explicit experimental-boundary assertions."""

from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import LOCKED_DEVELOPMENT_IDS, LOCKED_TEST_IDS


def allocate_profiles(
    profile_table: pd.DataFrame, test_fraction: float = 0.20, seed: int = 42
) -> tuple[pd.Series, pd.Series]:
    """Reproduce the historical target-stratified recording allocation.

    Parameters
    ----------
    profile_table : pandas.DataFrame
        Sorted table with one row per profile; columns ``profile_id`` and
        ``pm_max`` (deg C). Supplied ordering is preserved.
    test_fraction : float, default 0.20
        Fraction of profiles, not rows, reserved for final evaluation.
    seed : int, default 42
        Pseudorandom state used by scikit-learn.

    Returns
    -------
    development_ids, test_ids : pandas.Series
        Whole-profile allocations in scikit-learn's original returned order.

    Raises
    ------
    ValueError
        If columns are absent, profile IDs repeat, or tertiles cannot be made.

    Notes
    -----
    Three ``qcut`` bins use measured profile maxima, including final-test
    maxima for allocation. This is retrospective target-informed sampling;
    it is not a fully blind prospective holdout. No target enters features.
    """
    if not {"profile_id", "pm_max"} <= set(profile_table.columns):
        raise ValueError("Need profile_id and pm_max")
    if profile_table.profile_id.duplicated().any():
        raise ValueError("Each profile must have exactly one allocation row")
    strata = pd.qcut(profile_table["pm_max"], q=3, labels=["Low", "Medium", "High"])
    return train_test_split(
        profile_table["profile_id"],
        test_size=test_fraction,
        random_state=seed,
        stratify=strata,
    )


def assert_disjoint_profiles(
    training: Iterable[int], validation: Iterable[int], forbidden: Iterable[int] = ()
) -> None:
    """Reject shared sessions or a forbidden held-out training/validation ID.

    Parameters
    ----------
    training, validation, forbidden : iterable of int
        Recording identifiers; they are audit metadata, not predictors.

    Returns
    -------
    None
        Successful return certifies only ID isolation, not statistical independence.

    Raises
    ------
    ValueError
        If the two partitions overlap or either contains a forbidden ID.
    """
    tr, va, no = set(training), set(validation), set(forbidden)
    if tr & va or (tr | va) & no:
        raise ValueError("Profile isolation violated")


def assert_locked_partition(development: Iterable[int], test: Iterable[int]) -> None:
    """Verify the exact established 55/14 allocation without redefining it.

    Parameters
    ----------
    development, test : iterable of int
        Profile IDs obtained from the original dataset.

    Returns
    -------
    None
        Successful return confirms both exact sets.

    Raises
    ------
    ValueError
        If either assignment differs from the supplied scientific protocol.
    """
    if sorted(development) != LOCKED_DEVELOPMENT_IDS or sorted(test) != LOCKED_TEST_IDS:
        raise ValueError("The locked 55/14 scientific allocation changed")
    assert_disjoint_profiles(development, test)
