import numpy as np
import pandas as pd
import pytest
from pmsm_project.features import (
    make_paper_inputs,
    make_temporal_features,
    vectorised_temporal_sample,
    feature_names,
    feature_frame,
    stage6_feature_names,
)
from pmsm_project.config import BASE_COLS


def test_physical_formulas_and_exclusions():
    row = pd.DataFrame(
        dict(
            ambient=[20.0],
            coolant=[25.0],
            u_d=[3.0],
            u_q=[4.0],
            i_d=[6.0],
            i_q=[8.0],
            motor_speed=[60.0],
            pm=[999.0],
            profile_id=[71],
        )
    )
    f = make_paper_inputs(row)
    assert list(f) == BASE_COLS and "pm" not in f and "profile_id" not in f
    np.testing.assert_allclose(
        f[["u_s", "i_s", "S_el", "i_s_omega", "S_el_omega"]].iloc[0],
        [5.0, 10.0, 75.0, 20 * np.pi, 150 * np.pi],
        rtol=1e-14,
    )


def test_measured_column_required(profile):
    with pytest.raises(ValueError, match="Missing"):
        make_paper_inputs(profile.drop(columns="ambient"))


def test_weighted_population_moments_independent():
    x = np.array([10.0, 20.0, 30.0])
    f = pd.DataFrame({"signal": x})
    g = pd.Series([2] * 3)
    out = make_temporal_features(f, g, (3,))
    for t in range(3):
        w = 0.5 ** np.arange(t, -1, -1)
        mu = np.dot(w, x[: t + 1]) / w.sum()
        sd = np.sqrt(np.dot(w, (x[: t + 1] - mu) ** 2) / w.sum())
        np.testing.assert_allclose(out.signal_ewma_s3.iloc[t], mu, rtol=1e-14)
        np.testing.assert_allclose(out.signal_ewms_s3.iloc[t], sd, rtol=1e-14)


def test_profile_reset():
    x = pd.DataFrame({"signal": [10.0, 20.0, 100.0, 120.0]})
    g = pd.Series([2, 2, 3, 3])
    out = make_temporal_features(x, g, (3,))
    assert out.signal_ewma_s3.iloc[2] == 100 and out.signal_ewms_s3.iloc[2] == 0


def test_future_perturbation_cannot_change_past(profile):
    spans = (10, 50, 200, 1000)
    other = profile.copy()
    other.loc[200:, "i_q"] += 1e6
    a = feature_frame(profile, spans)
    b = feature_frame(other, spans)
    np.testing.assert_array_equal(a.iloc[:200], b.iloc[:200])
    assert not np.allclose(a.iloc[201:], b.iloc[201:])


def test_target_and_foreign_profile_do_not_enter_features(profile):
    spans = (10, 50, 200, 1000)
    a = feature_frame(profile, spans)
    changed = profile.copy()
    changed["pm"] += 100000
    np.testing.assert_array_equal(a, feature_frame(changed, spans))
    other = profile.copy()
    other["profile_id"] = 3
    other.index += len(profile)
    joined = pd.concat([profile, other])
    f = make_temporal_features(make_paper_inputs(joined), joined.profile_id, spans)
    np.testing.assert_allclose(a, f.loc[profile.index], rtol=0, atol=0)


def test_vectorised_column_order_and_values(profile):
    spans = (11, 57, 209, 1301)
    positions = np.array([0, 1, 2, 35, 128, 359])
    ref = make_temporal_features(make_paper_inputs(profile), profile.profile_id, spans)
    actual = vectorised_temporal_sample(make_paper_inputs(profile), positions, spans)
    assert list(actual) != list(ref)  # source sampler is intentionally span-major
    np.testing.assert_array_equal(actual[list(ref)], ref.iloc[positions])
    assert stage6_feature_names(BASE_COLS, spans) == feature_names(spans)
    np.testing.assert_array_equal(feature_frame(profile, spans), ref)


@pytest.mark.parametrize(
    "problem",
    [
        "misaligned",
        "duplicate_index",
        "nonchronological",
        "interleaved",
        "badspan",
        "duplicatespan",
        "nan",
        "type",
    ],
)
def test_temporal_invalid_inputs(profile, problem):
    x = make_paper_inputs(profile)
    g = profile.profile_id.copy()
    spans = (10, 50, 200, 1000)
    if problem == "misaligned":
        g.index = g.index + 1
    elif problem == "duplicate_index":
        x.index = [0] * len(x)
        g.index = x.index
    elif problem == "nonchronological":
        x = x.iloc[::-1]
        g = g.iloc[::-1]
    elif problem == "interleaved":
        g.iloc[10] = 3
    elif problem == "badspan":
        spans = (0,)
    elif problem == "duplicatespan":
        spans = (10, 10)
    elif problem == "nan":
        x.iloc[4, 0] = np.nan
    elif problem == "type":
        g = g.to_numpy()
    with pytest.raises((ValueError, TypeError)):
        make_temporal_features(x, g, spans)


def test_final_profile_contract(profile):
    mixed = profile.copy()
    mixed.loc[5, "profile_id"] = 3
    with pytest.raises(AssertionError):
        feature_frame(mixed, (10, 50, 200, 1000))
