import numpy as np
import pandas as pd
import pytest
from pmsm_project.sampling import nested_sample_order
from pmsm_project.validation import (
    allocate_profiles,
    assert_disjoint_profiles,
    assert_locked_partition,
)
from pmsm_project.config import LOCKED_DEVELOPMENT_IDS, LOCKED_TEST_IDS
from sklearn.model_selection import StratifiedKFold


def test_nested_rank_deterministic_endpoints():
    a = nested_sample_order(2500, 1280)
    b = nested_sample_order(2500, 1280)
    assert a == b and a[:2] == [0, 2499] and len(set(a)) == 1280
    for n in [80, 160, 320, 640]:
        assert set(a[:n]) < set(a[: 2 * n])


def test_nested_covers_all_positions():
    assert set(nested_sample_order(17, 17)) == set(range(17))


@pytest.mark.parametrize("length,budget", [(1, 2), (10, 1), (10, 11), (0, 0)])
def test_invalid_nested_budget(length, budget):
    with pytest.raises(ValueError):
        nested_sample_order(length, budget)


def test_split_rejects_overlap_and_forbidden():
    with pytest.raises(ValueError):
        assert_disjoint_profiles([2, 3], [3, 4])
    with pytest.raises(ValueError):
        assert_disjoint_profiles([2, 13], [3], [13])
    with pytest.raises(ValueError):
        assert_disjoint_profiles([2], [13], [13])
    assert_disjoint_profiles([2, 3], [4], [13])


def test_locked_allocation_not_just_disjoint():
    assert_locked_partition(LOCKED_DEVELOPMENT_IDS, LOCKED_TEST_IDS)
    with pytest.raises(ValueError):
        assert_locked_partition(LOCKED_DEVELOPMENT_IDS[:-1], LOCKED_TEST_IDS)


def test_duplicate_profile_allocation_rejected():
    with pytest.raises(ValueError):
        allocate_profiles(
            pd.DataFrame({"profile_id": [2, 2, 3], "pm_max": [20, 30, 40]})
        )
    with pytest.raises(ValueError):
        allocate_profiles(pd.DataFrame({"pm_max": [20, 30, 40]}))


@pytest.mark.integration
def test_original_split_and_grouped_folds(original_csv):
    f = pd.read_csv(original_csv, usecols=["profile_id", "pm"])
    table = (
        f.groupby("profile_id", sort=True)
        .agg(observations=("pm", "size"), pm_max=("pm", "max"))
        .reset_index()
    )
    dev, test = allocate_profiles(table)
    assert_locked_partition(dev, test)
    again = allocate_profiles(table)
    np.testing.assert_array_equal(dev, again[0])
    np.testing.assert_array_equal(test, again[1])
    cv = table.loc[table.profile_id.isin(dev)].reset_index(drop=True)
    strata = pd.qcut(cv.pm_max, 3, labels=["Low", "Medium", "High"])
    counts = []
    all_val = []
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=42).split(
        cv[["pm_max"]], strata
    ):
        assert_disjoint_profiles(cv.profile_id.iloc[tr], cv.profile_id.iloc[va], test)
        counts.append(int(cv.observations.iloc[va].sum()))
        all_val += cv.profile_id.iloc[va].tolist()
    assert counts == [401114, 362087, 291997]
    assert sorted(all_val) == LOCKED_DEVELOPMENT_IDS
