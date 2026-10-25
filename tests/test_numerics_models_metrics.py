import copy
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import lstsq
from pmsm_project.features import feature_frame
from pmsm_project.numerics import compress_profile_ols, qr_compress_profile, fit_qr_ols
from pmsm_project.metrics import score
from pmsm_project.models import build_pipeline, make_regressor
from pathlib import Path
import json


def fixture_configurations():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "provenance/historical_configurations.json"
        ).read_text()
    )


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def test_augmented_qr_independent_sse(profile):
    spans = (10, 50, 200, 1000)
    x = feature_frame(profile, spans).to_numpy()
    y = profile.pm.to_numpy()
    rng = np.random.default_rng(573)
    w = rng.normal(size=110)
    w[-1] = -1
    direct = np.linalg.norm(np.column_stack([np.ones(len(x)), x, y]) @ w) ** 2
    for result in [
        compress_profile_ols(profile, spans, 128),
        qr_compress_profile(profile, spans, 128),
    ]:
        np.testing.assert_allclose(
            np.linalg.norm(result["R"] @ w) ** 2, direct, rtol=1e-11, atol=1e-7
        )
        np.testing.assert_allclose(result["mean"], x.mean(0), rtol=1e-13, atol=1e-10)
        np.testing.assert_allclose(result["var"], x.var(0), rtol=1e-13, atol=1e-9)


def test_qr_predictions_against_direct_scaled_solver(profile, tmp_path):
    spans = (10, 50, 200, 1000)
    f = profile.copy()
    g = profile.copy()
    g.index += len(f)
    g["profile_id"] = 3
    g["ambient"] += 3
    g["i_q"] *= 1.2
    g["pm"] += 2
    both = pd.concat([f, g])
    bundle = fit_qr_ols(both, spans)
    x = np.vstack([feature_frame(f, spans), feature_frame(g, spans)])
    y = both.pm.to_numpy()
    scale = StandardScaler()
    xs = scale.fit_transform(x)
    coef = lstsq(np.column_stack([np.ones(len(x)), xs]), y, lapack_driver="gelsd")[0]
    prediction = np.column_stack([np.ones(len(x)), xs]) @ coef
    actual = bundle["intercept"] + x @ bundle["coef"]
    # These synthetic thermal moments are nearly collinear: compare fitted
    # predictions, not ill-conditioned individual coefficients or QR signs.
    np.testing.assert_allclose(actual, prediction, rtol=1e-7, atol=2e-5)
    assert bundle["train_rows"] == 720 and np.isfinite(actual).all()
    path = tmp_path / "qr_ols.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(actual, restored["intercept"] + x @ restored["coef"])


def test_numerical_contract_errors(profile):
    with pytest.raises(ValueError):
        compress_profile_ols(profile, (10, 10, 200, 1000))
    with pytest.raises(ValueError):
        compress_profile_ols(profile, (10, 50, 200, 1000), 1)
    with pytest.raises(ValueError):
        compress_profile_ols(profile.iloc[::-1], (10, 50, 200, 1000))


def test_metrics_independent_and_bias_variance():
    y = np.array([20.0, 24.0, 28.0, 32.0])
    yp = np.array([21.0, 22.0, 32.0, 31.0])
    e = yp - y
    s = score(y, yp)
    assert s["MSE_C2"] == 5.5 and s["Bias_C"] == 0.5 and s["MAE_C"] == 2
    np.testing.assert_allclose(s["RMSE_C"] ** 2, s["Bias_C"] ** 2 + e.var(), rtol=1e-14)
    np.testing.assert_allclose(
        [s["MSE_C2"], s["MAE_C"], s["R2"]],
        [mean_squared_error(y, yp), mean_absolute_error(y, yp), r2_score(y, yp)],
    )
    assert s["Max_AE_C"] == 4 and s["P95_AE_C"] == pytest.approx(3.7)


def test_constant_target_r2_is_nan():
    assert np.isnan(score(np.ones(3), np.ones(3))["R2"])


@pytest.mark.parametrize(
    "y,p",
    [([], []), ([1, 2], [1]), ([[1]], [[1]]), ([1, np.nan], [1, 2]), ([np.inf], [1])],
)
def test_invalid_metrics(y, p):
    with pytest.raises(ValueError):
        score(y, p)


@pytest.mark.parametrize(
    "name", ["k-NN", "Random Forest", "Extra Trees", "MLP", "LinearSVR"]
)
def test_final_factory_roundtrip_and_train_only_scaling(name, tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(300, 108))
    y = 40 + 2 * x[:, 0] - x[:, 1]
    model = build_pipeline(name, fixture_configurations())
    model.fit(x, y)
    prediction = model.predict(x[:7] + 0.1)
    assert prediction.shape == (7,) and np.isfinite(prediction).all()
    np.testing.assert_allclose(
        model.regressor_.named_steps["scaler"].mean_, x.mean(0), atol=1e-14
    )
    np.testing.assert_allclose(model.transformer_.mean_, [y.mean()], atol=1e-14)
    p = tmp_path / f"{name}.joblib"
    joblib.dump(model, p)
    # Parallel tree summation can change the last bit after reload (observed
    # 7.1e-15 deg C). This is far below 1e-12 deg C and is not model drift.
    np.testing.assert_allclose(
        prediction, joblib.load(p).predict(x[:7] + 0.1), rtol=1e-13, atol=1e-12
    )


def test_fixture_copy_and_distinct_mlp_protocols():
    a = fixture_configurations()
    a["MLP"]["spans"][0] = 10000
    assert fixture_configurations()["MLP"]["spans"][0] == 52
    hp = fixture_configurations()["MLP"]["hyperparameters"]
    d = make_regressor("MLP", hp)
    f = build_pipeline("MLP", fixture_configurations()).regressor.named_steps["model"]
    assert d.max_iter == 200 and not d.early_stopping
    assert f.max_iter == 400 and f.early_stopping and f.n_iter_no_change == 15


def test_model_unknown_and_determinism():
    with pytest.raises(ValueError):
        make_regressor("unknown", {})
    with pytest.raises((ValueError, KeyError)):
        build_pipeline("unknown", fixture_configurations())
    rng = np.random.default_rng(42)
    x = rng.normal(size=(90, 108))
    y = 40 + x[:, 0]
    a = build_pipeline("Extra Trees", fixture_configurations()).fit(x, y)
    b = build_pipeline("Extra Trees", fixture_configurations()).fit(x, y)
    np.testing.assert_allclose(a.predict(x), b.predict(x), rtol=1e-13, atol=1e-12)
