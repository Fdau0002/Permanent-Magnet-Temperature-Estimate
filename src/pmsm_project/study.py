"""Recovered development objectives with explicit development-only state.

The numerical statements come from the supplied historical reconstruction;
provenance/objective_extraction.json records the source cells. This module
does not read a saved trial file or a historical optimum.
"""

from __future__ import annotations
import time, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import lstsq
from .features import make_paper_inputs, make_temporal_features
from .sampling import nested_sample_order
from .numerics import compress_profile_ols
from .models import make_regressor
from .search import MODEL_SPACES, TRAIN_SAMPLES, VALIDATION_SAMPLES, decode_spans
from .config import LOCKED_DEVELOPMENT_IDS, LOCKED_TEST_IDS


class DevelopmentStudy:
    """Bind unchanged numerical objectives to a verified development partition.

    Parameters
    ----------
    development : pandas.DataFrame
        Chronological raw measurements; target ``pm`` in deg C.
    profile_cv : pandas.DataFrame
        One row per profile with ``profile_id`` and integer ``cv_fold`` (1--3).
    strict : bool, default True
        Enforce original 55 profiles and 1,055,198 rows. False is exclusively
        for labelled synthetic numerical tests, never production selection.

    Raises
    ------
    ValueError
        On overlapping held-out IDs, missing folds or incompatible partitions.

    Examples
    --------
    >>> # study = DevelopmentStudy(development, profile_cv)
    >>> # mean_mse, fold_mse = study.evaluate_ols_cv([10, 50, 200, 1000])
    """

    def __init__(self, development, profile_cv, strict=True):
        ids = set(map(int, development.profile_id))
        if ids & set(LOCKED_TEST_IDS):
            raise ValueError(
                "Held-out profiles are forbidden in development objectives"
            )
        if (
            profile_cv.profile_id.duplicated().any()
            or ids != set(profile_cv.profile_id)
            or set(profile_cv.cv_fold) != {1, 2, 3}
        ):
            raise ValueError(
                "Require exactly one of three CV folds for every development profile"
            )
        if strict and (
            sorted(ids) != LOCKED_DEVELOPMENT_IDS or len(development) != 1055198
        ):
            raise ValueError(
                "Production optimisation requires all original development profiles/rows"
            )
        self.development = development
        self.profile_cv = profile_cv.copy()
        self.fold_lookup = (
            profile_cv.set_index("profile_id")["cv_fold"].astype(int).to_dict()
        )
        self.samples_per_profile = 80
        self.train_samples = TRAIN_SAMPLES.copy()
        self.validation_samples = VALIDATION_SAMPLES

    def build_candidate_features(self, spans):
        """Construct features for candidate EWMA/EWMS spans.

        Full original 2-Hz histories are processed before
        selecting 80 observations per development profile.

        Parameters
        ----------
        spans : sequence[int]
            Four ordered causal exponential spans in samples (2 Hz source).

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Sampled predictors (mixed physical units), target (deg C), profile IDs.

        Raises
        ------
        ValueError
            If candidate parameters, profile lengths or estimator settings are invalid.
        AssertionError
            If finite-value, expected-size or profile-isolation invariants fail.

        Notes
        -----
        Source arithmetic is retained. Features use complete chronological profiles
        before sampling; all scalers are fitted within the fitting fold. Fold MSE
        is in original temperature units after inverse transformation. QR solves
        can be rank deficient; fitted rank is recorded instead of forming inverses.
        """
        feature_arrays = []
        target_arrays = []
        profile_arrays = []
        for pid, profile in self.development.groupby("profile_id", sort=False):
            X_original = make_paper_inputs(profile)
            X_temporal = make_temporal_features(
                X_original, profile["profile_id"], spans
            )
            n = min(self.samples_per_profile, len(profile))
            positions = np.linspace(0, len(profile) - 1, n, dtype=int)
            feature_arrays.append(X_temporal.iloc[positions].to_numpy(dtype=np.float64))
            target_arrays.append(profile["pm"].iloc[positions].to_numpy())
            profile_arrays.append(np.full(n, pid))
        X = np.vstack(feature_arrays)
        y = np.concatenate(target_arrays)
        profile_ids = np.concatenate(profile_arrays)
        assert X.shape[1] == 108
        assert len(X) == 4400
        assert np.isfinite(X).all()
        return (X, y, profile_ids)

    def evaluate_ols_cv(self, spans):
        """Evaluate a four-span configuration with the same
        fixed, grouped, stratified 3-fold CV.

        Return average CV MSE and individual fold scores.

        Parameters
        ----------
        spans : sequence[int]
            Four ordered causal exponential spans in samples (2 Hz source).

        Returns
        -------
        tuple[float, list[float]]
            Arithmetic mean and three fold MSE scores (deg C squared).

        Raises
        ------
        ValueError
            If candidate parameters, profile lengths or estimator settings are invalid.
        AssertionError
            If finite-value, expected-size or profile-isolation invariants fail.

        Notes
        -----
        Source arithmetic is retained. Features use complete chronological profiles
        before sampling; all scalers are fitted within the fitting fold. Fold MSE
        is in original temperature units after inverse transformation. QR solves
        can be rank deficient; fitted rank is recorded instead of forming inverses.
        """
        X, y, profile_ids = self.build_candidate_features(spans)
        fold_ids = pd.Series(profile_ids).map(self.fold_lookup).to_numpy(dtype=int)
        fold_mse = []
        for fold in [1, 2, 3]:
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            assert train_mask.sum() > 0
            assert val_mask.sum() > 0
            assert set(profile_ids[train_mask]).isdisjoint(set(profile_ids[val_mask]))
            pipeline = Pipeline(
                [("scaler", StandardScaler()), ("model", LinearRegression())]
            )
            pipeline.fit(X[train_mask], y[train_mask])
            predictions = pipeline.predict(X[val_mask])
            mse = mean_squared_error(y[val_mask], predictions)
            fold_mse.append(float(mse))
        return (float(np.mean(fold_mse)), fold_mse)

    def evaluate_model_cv(self, model_name, parameters):
        """Evaluate model hyperparameters on the fixed
        stratified grouped 3-fold cross-validation split.

        Parameters
        ----------
        model_name : str
            SVR, k-NN, Random Forest, Extra Trees or MLP.
        parameters : sequence
            First span, three gaps, then model parameters in search-space order.

        Returns
        -------
        dict
            Spans, hyperparameters, fold MSE, mean MSE and convergence count.

        Raises
        ------
        ValueError
            If candidate parameters, profile lengths or estimator settings are invalid.
        AssertionError
            If finite-value, expected-size or profile-isolation invariants fail.

        Notes
        -----
        Source arithmetic is retained. Features use complete chronological profiles
        before sampling; all scalers are fitted within the fitting fold. Fold MSE
        is in original temperature units after inverse transformation. QR solves
        can be rank deficient; fitted rank is recorded instead of forming inverses.
        """
        spans = decode_spans(parameters[:4])
        hp = {
            dimension.name: value
            for dimension, value in zip(MODEL_SPACES[model_name], parameters[4:])
        }
        X, y, profile_ids = self.build_candidate_features(spans)
        fold_lookup = self.profile_cv.set_index("profile_id")["cv_fold"]
        fold_ids = pd.Series(profile_ids).map(self.fold_lookup).to_numpy()
        assert not pd.isna(fold_ids).any()
        fold_scores = []
        convergence_warnings = 0
        for fold in [1, 2, 3]:
            training_mask = fold_ids != fold
            validation_mask = fold_ids == fold
            assert set(profile_ids[training_mask]).isdisjoint(
                set(profile_ids[validation_mask])
            )
            estimator = make_regressor(model_name, hp)
            feature_pipeline = Pipeline(
                [("scaler", StandardScaler()), ("model", estimator)]
            )
            pipeline = TransformedTargetRegressor(
                regressor=feature_pipeline, transformer=StandardScaler()
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(X[training_mask], y[training_mask])
            convergence_warnings += sum(
                (issubclass(w.category, ConvergenceWarning) for w in caught)
            )
            predictions = pipeline.predict(X[validation_mask])
            if not np.isfinite(predictions).all():
                raise ValueError(f"{model_name} produced invalid predictions")
            mse = mean_squared_error(y[validation_mask], predictions)
            fold_scores.append(float(mse))
        return {
            "spans": spans,
            "hyperparameters": hp,
            "fold_scores": fold_scores,
            "mean_cv_mse": float(np.mean(fold_scores)),
            "convergence_warnings": convergence_warnings,
        }

    def evaluate_full_ols_cv(self, spans):
        """Evaluate an OLS configuration on all development rows.

        Uses profile-wise QR compression and training-fold-only
        standardisation. No test profiles are accessed.

        Parameters
        ----------
        spans : sequence[int]
            Four ordered causal exponential spans in samples (2 Hz source).

        Returns
        -------
        dict
            Spans, fold table with numerical rank/MSE, and measured timing (seconds).

        Raises
        ------
        ValueError
            If candidate parameters, profile lengths or estimator settings are invalid.
        AssertionError
            If finite-value, expected-size or profile-isolation invariants fail.

        Notes
        -----
        Source arithmetic is retained. Features use complete chronological profiles
        before sampling; all scalers are fitted within the fitting fold. Fold MSE
        is in original temperature units after inverse transformation. QR solves
        can be rank deficient; fitted rank is recorded instead of forming inverses.
        """
        start_total = time.perf_counter()
        profile_stats = {}
        for count, (pid, profile) in enumerate(
            self.development.groupby("profile_id", sort=False), start=1
        ):
            pid = int(pid)
            profile_stats[pid] = compress_profile_ols(profile, spans)
            if count % 10 == 0 or count == 55:
                print(f"Processed {count}/55 profiles", flush=True)
        feature_time = time.perf_counter() - start_total
        start_cv = time.perf_counter()
        fold_results = []
        for fold in [1, 2, 3]:
            training_ids = [
                pid for pid in profile_stats if self.fold_lookup[pid] != fold
            ]
            validation_ids = [
                pid for pid in profile_stats if self.fold_lookup[pid] == fold
            ]
            assert set(training_ids).isdisjoint(set(validation_ids))
            n_train = sum((profile_stats[pid]["n"] for pid in training_ids))
            mu = (
                sum(
                    (
                        profile_stats[pid]["n"] * profile_stats[pid]["mean"]
                        for pid in training_ids
                    )
                )
                / n_train
            )
            variance = (
                sum(
                    (
                        profile_stats[pid]["n"]
                        * (
                            profile_stats[pid]["var"]
                            + (profile_stats[pid]["mean"] - mu) ** 2
                        )
                        for pid in training_ids
                    )
                )
                / n_train
            )
            scale = np.sqrt(np.maximum(variance, 0))
            scale[scale == 0] = 1.0
            R_train = np.vstack([profile_stats[pid]["R"] for pid in training_ids])
            R_scaled = R_train.copy()
            R_scaled[:, 1:-1] = (R_train[:, 1:-1] - R_train[:, [0]] * mu) / scale
            coefficients, _, rank, singular_values = lstsq(
                R_scaled[:, :-1], R_scaled[:, -1], lapack_driver="gelsd"
            )
            beta = coefficients[1:] / scale
            intercept = coefficients[0] - np.dot(mu, beta)
            residual_vector = np.concatenate([[intercept], beta, [-1.0]])
            n_validation = 0
            validation_sse = 0.0
            for pid in validation_ids:
                R = profile_stats[pid]["R"]
                residuals_compressed = R @ residual_vector
                validation_sse += float(
                    np.dot(residuals_compressed, residuals_compressed)
                )
                n_validation += profile_stats[pid]["n"]
            mse = validation_sse / n_validation
            assert np.isfinite(mse)
            assert mse >= 0
            fold_results.append(
                {
                    "Fold": fold,
                    "Training_Rows": n_train,
                    "Validation_Rows": n_validation,
                    "MSE": mse,
                    "Numerical_Rank": rank,
                }
            )
        cv_time = time.perf_counter() - start_cv
        results = pd.DataFrame(fold_results)
        mean_cv_mse = results["MSE"].mean()
        return {
            "spans": list(map(int, spans)),
            "fold_results": results,
            "mean_cv_mse": float(mean_cv_mse),
            "feature_time_s": feature_time,
            "cv_time_s": cv_time,
            "total_time_s": time.perf_counter() - start_total,
        }

    def evaluate_larger_model_cv(self, model_name, parameters):
        """Evaluate one Bayesian candidate using:
        - Full original 2-Hz temporal histories
        - Nested training samples
        - Fixed 640-sample/profile validation
        - Three profile-grouped CV folds

        Parameters
        ----------
        model_name : str
            SVR, k-NN, Random Forest, Extra Trees or MLP.
        parameters : sequence
            First span, three gaps, then model parameters in search-space order.

        Returns
        -------
        dict
            Selected candidate, three MSE values, timing and convergence count.

        Raises
        ------
        ValueError
            If candidate parameters, profile lengths or estimator settings are invalid.
        AssertionError
            If finite-value, expected-size or profile-isolation invariants fail.

        Notes
        -----
        Source arithmetic is retained. Features use complete chronological profiles
        before sampling; all scalers are fitted within the fitting fold. Fold MSE
        is in original temperature units after inverse transformation. QR solves
        can be rank deficient; fitted rank is recorded instead of forming inverses.
        """
        start_total = time.perf_counter()
        n_train = self.train_samples[model_name]
        n_val = self.validation_samples
        n_select = max(n_train, n_val)
        spans = decode_spans(parameters[:4])
        hp = {
            dimension.name: value
            for dimension, value in zip(MODEL_SPACES[model_name], parameters[4:])
        }
        feature_arrays = []
        target_arrays = []
        profile_arrays = []
        rank_arrays = []
        for pid, profile in self.development.groupby("profile_id", sort=False):
            order = nested_sample_order(len(profile), n_select)
            positions = np.sort(np.asarray(order, dtype=int))
            rank_lookup = {int(position): rank for rank, position in enumerate(order)}
            ranks = np.asarray([rank_lookup[int(position)] for position in positions])
            X_original = make_paper_inputs(profile)
            X_full = make_temporal_features(X_original, profile["profile_id"], spans)
            X_selected = X_full.iloc[positions].to_numpy(dtype=np.float64)
            y_selected = profile["pm"].iloc[positions].to_numpy(dtype=np.float64)
            feature_arrays.append(X_selected)
            target_arrays.append(y_selected)
            profile_arrays.append(np.full(len(positions), int(pid)))
            rank_arrays.append(ranks)
        X = np.vstack(feature_arrays)
        y = np.concatenate(target_arrays)
        profile_ids = np.concatenate(profile_arrays)
        sample_ranks = np.concatenate(rank_arrays)
        feature_time = time.perf_counter() - start_total
        assert X.shape[1] == 108
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()
        fold_ids = pd.Series(profile_ids).map(self.fold_lookup).to_numpy(dtype=int)
        fold_scores = []
        convergence_warnings = 0
        training_time = 0.0
        inference_time = 0.0
        for fold in [1, 2, 3]:
            train_mask = (fold_ids != fold) & (sample_ranks < n_train)
            val_mask = (fold_ids == fold) & (sample_ranks < n_val)
            assert set(profile_ids[train_mask]).isdisjoint(set(profile_ids[val_mask]))
            estimator = make_regressor(model_name, hp)
            if model_name == "MLP":
                estimator.max_iter = 400
                estimator.early_stopping = True
                estimator.validation_fraction = 0.1
                estimator.n_iter_no_change = 15
            pipeline = TransformedTargetRegressor(
                regressor=Pipeline(
                    [("scaler", StandardScaler()), ("model", estimator)]
                ),
                transformer=StandardScaler(),
            )
            start_fit = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(X[train_mask], y[train_mask])
            training_time += time.perf_counter() - start_fit
            convergence_warnings += sum(
                (issubclass(w.category, ConvergenceWarning) for w in caught)
            )
            start_predict = time.perf_counter()
            predictions = pipeline.predict(X[val_mask])
            inference_time += time.perf_counter() - start_predict
            assert np.isfinite(predictions).all()
            fold_scores.append(float(mean_squared_error(y[val_mask], predictions)))
        mean_mse = float(np.mean(fold_scores))
        return {
            "spans": spans,
            "hyperparameters": hp,
            "fold_scores": fold_scores,
            "mean_cv_mse": mean_mse,
            "feature_time_s": feature_time,
            "training_time_s": training_time,
            "inference_time_s": inference_time,
            "total_time_s": time.perf_counter() - start_total,
            "convergence_warnings": convergence_warnings,
        }
