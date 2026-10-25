"""Execute original uploaded function definitions independently as regression oracles.

These tests supplement (not replace) independent mathematical identity tests.
They ensure refactoring preserves the supplied algorithms on valid measurements.
"""

import ast
import hashlib
import json
from pathlib import Path
import time
import heapq
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import qr, lstsq
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from pmsm_project.features import (
    make_paper_inputs,
    make_temporal_features,
    vectorised_temporal_sample,
    stage6_feature_names,
    feature_frame,
    feature_names,
)
from pmsm_project.numerics import compress_profile_ols
from pmsm_project.sampling import nested_sample_order
from pmsm_project.models import make_regressor

ROOT = Path(__file__).resolve().parents[1]


def original_function(cell, name):
    nb = json.loads(
        (ROOT / "provenance/source_notebooks/stage9_original.ipynb").read_text()
    )
    node = next(
        n
        for n in ast.parse("".join(nb["cells"][cell]["source"])).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    ns = dict(
        np=np,
        pd=pd,
        time=time,
        heapq=heapq,
        qr=qr,
        lstsq=lstsq,
        make_paper_inputs=make_paper_inputs,
        make_temporal_features=make_temporal_features,
        vectorised_temporal_sample=vectorised_temporal_sample,
        stage6_feature_names=stage6_feature_names,
        SVR=SVR,
        KNeighborsRegressor=KNeighborsRegressor,
        RandomForestRegressor=RandomForestRegressor,
        ExtraTreesRegressor=ExtraTreesRegressor,
        MLPRegressor=MLPRegressor,
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "original_upload", "exec"), ns
    )
    return ns[name]


def test_original_base_temporal_and_vectorised_exact(profile):
    spans = (91, 358, 1712, 7591)
    x = make_paper_inputs(profile)
    pd.testing.assert_frame_equal(
        x, original_function(17, "make_paper_inputs")(profile), check_exact=True
    )
    pd.testing.assert_frame_equal(
        make_temporal_features(x, profile.profile_id, spans),
        original_function(21, "make_temporal_features")(x, profile.profile_id, spans),
        check_exact=True,
    )
    pos = np.array([0, 1, 33, 180, 359])
    pd.testing.assert_frame_equal(
        vectorised_temporal_sample(x, pos, spans),
        original_function(66, "vectorised_temporal_sample")(x, pos, spans),
        check_exact=True,
    )


def test_original_nested_and_qr_exact(profile):
    assert nested_sample_order(2500, 1280) == original_function(
        64, "nested_sample_order"
    )(2500, 1280)
    a = compress_profile_ols(profile, (10, 50, 200, 1000), 128)
    b = original_function(79, "compress_profile_ols")(profile, (10, 50, 200, 1000), 128)
    assert a["n"] == b["n"]
    for k in ["mean", "var", "R"]:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize(
    "name,hp",
    [
        ("SVR", {"C": 0.15, "epsilon": 0.08}),
        ("k-NN", {"n_neighbors": 54, "weights": "distance"}),
        (
            "Random Forest",
            {
                "n_estimators": 87,
                "max_depth": 30,
                "min_samples_split": 6,
                "min_samples_leaf": 1,
                "bootstrap": "yes",
            },
        ),
        (
            "Extra Trees",
            {
                "n_estimators": 87,
                "max_depth": 26,
                "min_samples_split": 4,
                "min_samples_leaf": 1,
                "bootstrap": "yes",
            },
        ),
        (
            "MLP",
            {
                "n_layers": 1,
                "n_units": 4,
                "alpha": 0.0072,
                "learning_rate_init": 0.0035,
            },
        ),
    ],
)
def test_development_model_all_parameters_match_original(name, hp):
    assert (
        make_regressor(name, hp).get_params()
        == original_function(59, "make_regressor")(name, hp).get_params()
    )
