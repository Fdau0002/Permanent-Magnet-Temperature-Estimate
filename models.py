"""Explicit historical development and final estimator factories.

Development MLP: 200 epochs maximum, no early stopping. Final MLP: 400
epochs maximum with row-based internal early stopping. These are distinct
provided procedures and are intentionally preserved, not reconciled.
"""

from __future__ import annotations
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from .config import SEED


def make_regressor(model_name: str, hp: dict):
    """Construct the original development-stage unscaled base estimator.

    Parameters
    ----------
    model_name : str
        SVR, k-NN, Random Forest, Extra Trees or MLP.
    hp : dict
        Candidate parameters from the unchanged skopt search spaces. C and
        epsilon are defined in standardised target units in scaled experiments.

    Returns
    -------
    sklearn.base.RegressorMixin
        Unfitted estimator with the supplied architecture, solver and seed 42.

    Raises
    ------
    ValueError
        If the model family is unsupported.
    KeyError
        If an expected hyperparameter is missing.

    Notes
    -----
    This factory does not run Bayesian optimisation. Its surrounding notebook
    objective fits fold-local scalers. Development MLP has early_stopping=False
    and max_iter=200, as supplied; this differs from final training."""
    if model_name == "SVR":
        return SVR(
            kernel="rbf", C=float(hp["C"]), epsilon=float(hp["epsilon"]), gamma="scale"
        )
    if model_name == "k-NN":
        return KNeighborsRegressor(
            n_neighbors=int(hp["n_neighbors"]), weights=hp["weights"], n_jobs=2
        )
    if model_name in ("Random Forest", "Extra Trees"):
        cls = (
            RandomForestRegressor
            if model_name == "Random Forest"
            else ExtraTreesRegressor
        )
        return cls(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            min_samples_split=int(hp["min_samples_split"]),
            min_samples_leaf=int(hp["min_samples_leaf"]),
            bootstrap=(hp["bootstrap"] == "yes"),
            max_features="sqrt",
            random_state=42,
            n_jobs=2,
        )
    if model_name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(int(hp["n_units"]),) * int(hp["n_layers"]),
            alpha=float(hp["alpha"]),
            activation="relu",
            solver="adam",
            learning_rate_init=float(hp["learning_rate_init"]),
            batch_size=64,
            max_iter=200,
            random_state=42,
            early_stopping=False,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(name: str, configurations: dict) -> TransformedTargetRegressor:
    """Construct the original frozen final scaled estimator (except QR OLS).

    Parameters
    ----------
    name : str
        k-NN, Random Forest, Extra Trees, MLP or LinearSVR.
    configurations : dict
        Required validated manifest model mapping; there is no historical fallback.
        This argument supports audit fixtures; it never tunes on test targets.

    Returns
    -------
    sklearn.compose.TransformedTargetRegressor
        A training-only StandardScaler for X, original estimator, and a
        training-only StandardScaler for target temperature (deg C).

    Raises
    ------
    ValueError
        If OLS or another unsupported name is supplied; OLS uses fit_qr_ols.
    KeyError
        If model configuration keys are absent.

    Notes
    -----
    Final MLP retains internal row-based early stopping (10% of development
    rows, patience 15, max_iter=400). It is not internally profile-disjoint.
    LinearSVR retains epsilon_insensitive loss, dual=True, tol=1e-4 and
    max_iter=10000. Estimator prediction returns degrees Celsius after inverse
    transformation. No trained artifact is restored by this constructor."""
    hp = configurations[name]["hyperparameters"]
    if name == "k-NN":
        estimator = KNeighborsRegressor(
            n_neighbors=hp["n_neighbors"], weights=hp["weights"], n_jobs=2
        )
    elif name in ("Random Forest", "Extra Trees"):
        cls = RandomForestRegressor if name == "Random Forest" else ExtraTreesRegressor
        estimator = cls(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"],
            min_samples_leaf=hp["min_samples_leaf"],
            bootstrap=hp["bootstrap"] == "yes",
            max_features="sqrt",
            random_state=SEED,
            n_jobs=2,
        )
    elif name == "MLP":
        estimator = MLPRegressor(
            hidden_layer_sizes=(hp["n_units"],) * hp["n_layers"],
            activation="relu",
            solver="adam",
            alpha=hp["alpha"],
            learning_rate_init=hp["learning_rate_init"],
            batch_size=64,
            max_iter=400,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=15,
        )
    elif name == "LinearSVR":
        estimator = LinearSVR(
            C=hp["C"],
            epsilon=hp["epsilon"],
            loss=hp["loss"],
            dual=hp["dual"],
            tol=hp["tol"],
            max_iter=hp["max_iter"],
            random_state=SEED,
        )
    else:
        raise ValueError("Unknown model: " + name)
    return TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()), ("model", estimator)]),
        transformer=StandardScaler(),
    )
