"""Original GP search spaces and explicit experiment budgets.

Spans are samples at 2 Hz. C and epsilon use standardised target units.
No winner or historical score is present in this module.
"""

from skopt.space import Integer, Real, Categorical

SPAN_SPACE = [
    Integer(5, 100, name="s1"),
    Integer(1, 400, name="gap_2"),
    Integer(1, 1500, name="gap_3"),
    Integer(1, 6000, name="gap_4"),
]

MODEL_SPACES = {
    "SVR": [
        Real(0.001, 10, prior="log-uniform", name="C"),
        Real(0.01, 0.3, prior="log-uniform", name="epsilon"),
    ],
    "k-NN": [
        Integer(1, 512, name="n_neighbors"),
        Categorical(["uniform", "distance"], name="weights"),
    ],
    "Random Forest": [
        Integer(20, 100, name="n_estimators"),
        Integer(5, 30, name="max_depth"),
        Integer(2, 20, name="min_samples_split"),
        Integer(1, 10, name="min_samples_leaf"),
        Categorical(["yes", "no"], name="bootstrap"),
    ],
    "Extra Trees": [
        Integer(20, 100, name="n_estimators"),
        Integer(5, 30, name="max_depth"),
        Integer(2, 20, name="min_samples_split"),
        Integer(1, 10, name="min_samples_leaf"),
        Categorical(["yes", "no"], name="bootstrap"),
    ],
    "MLP": [
        Integer(1, 3, name="n_layers"),
        Integer(4, 32, name="n_units"),
        Real(1e-07, 0.01, prior="log-uniform", name="alpha"),
        Real(0.0001, 0.01, prior="log-uniform", name="learning_rate_init"),
    ],
}


def decode_spans(parameters):
    """Convert the first span and three gaps into four ordered spans.

    Parameters
    ----------
    parameters : sequence of int, length 4
        First span followed by three positive increments, in source samples.

    Returns
    -------
    list of int
        Cumulative spans in samples at 2 Hz, in increasing order when gaps
        are positive. The unchanged search space enforces positive gaps.

    Raises
    ------
    ValueError, TypeError
        If four integer-convertible values are not supplied.

    Examples
    --------
    >>> decode_spans([10, 40, 150, 800])
    [10, 50, 200, 1000]
    """
    a, d2, d3, d4 = map(int, parameters)
    return [a, a + d2, a + d2 + d3, a + d2 + d3 + d4]


def encode_spans(spans):
    """Convert four ordered spans into the original first-span/gap space.

    Parameters
    ----------
    spans : sequence of int, length 4
        Strictly increasing causal-memory spans, in 2-Hz source samples.

    Returns
    -------
    list of int
        First span and the three consecutive positive differences.

    Raises
    ------
    ValueError, TypeError
        If four integer-convertible values are not supplied or ordering fails.

    Examples
    --------
    >>> encode_spans([10, 50, 200, 1000])
    [10, 40, 150, 800]
    """
    a, b, c, d = map(int, spans)
    if not a < b < c < d:
        raise ValueError("Spans must be strictly increasing")
    return [a, b - a, c - b, d - c]


TRAIN_SAMPLES = {
    "SVR": 320,
    "k-NN": 640,
    "Random Forest": 640,
    "Extra Trees": 640,
    "MLP": 640,
}
VALIDATION_SAMPLES = 640


def protocol():
    """Return the immutable production search contract for manifest comparison.

    Returns
    -------
    dict
        Dimension bounds/priors, trial counts, acquisition settings and budgets.

    Notes
    -----
    Counts include freshly rescored seeds in the full OLS and scaled searches.
    Initial OLS pilot and expanded OLS retain skopt's default acquisition solver.
    """

    def describe(d):
        if isinstance(d, Categorical):
            return {
                "name": d.name,
                "type": "Categorical",
                "categories": list(d.categories),
            }
        return {
            "name": d.name,
            "type": type(d).__name__,
            "low": int(d.low) if isinstance(d, Integer) else float(d.low),
            "high": int(d.high) if isinstance(d, Integer) else float(d.high),
            "prior": d.prior,
        }

    return {
        "seed": 42,
        "objective": "unweighted mean of three profile-grouped fold MSE values in deg C squared",
        "pilot_ols": {
            "algorithm": "gp_minimize",
            "trials": 8,
            "initial_points": 4,
            "acquisition": "EI",
            "train_per_profile": 80,
            "dimensions": [
                describe(d)
                for d in [
                    Integer(5, 30, name="span_1"),
                    Integer(31, 150, name="span_2"),
                    Integer(151, 700, name="span_3"),
                    Integer(701, 3000, name="span_4"),
                ]
            ],
        },
        "expanded_ols": {
            "algorithm": "Optimizer GP ask/tell",
            "new_trials": 20,
            "warm_start_trials": 8,
            "initial_points": 10,
            "acquisition": "EI",
            "train_per_profile": 80,
        },
        "pilot_models": {
            "algorithm": "Optimizer GP ask/tell",
            "trials_each": 8,
            "initial_points": 5,
            "acquisition": "EI",
            "acquisition_optimizer": "sampling",
            "train_per_profile": 80,
        },
        "full_ols": {
            "algorithm": "Optimizer GP ask/tell",
            "trials": 20,
            "fresh_seed_evaluations": 1,
            "initial_points": 6,
            "acquisition": "EI",
            "acquisition_optimizer": "sampling",
            "training": "all development rows; augmented profile QR; gelsd",
        },
        "scaled_models": {
            "algorithm": "Optimizer GP ask/tell",
            "trials_each": 12,
            "fresh_seed_evaluations": 1,
            "initial_points": 5,
            "acquisition": "EI",
            "acquisition_optimizer": "sampling",
            "train_per_profile": TRAIN_SAMPLES,
            "validation_per_profile": 640,
        },
        "span_dimensions": [describe(d) for d in SPAN_SPACE],
        "model_dimensions": {
            name: [describe(d) for d in dims] for name, dims in MODEL_SPACES.items()
        },
        "stage5": {
            "training_per_profile": [80, 160, 320, 640],
            "validation_per_profile": 640,
            "repeats": 3,
        },
        "stage9": {
            "samples_per_profile": 640,
            "expanded_train_per_profile": 1280,
            "expanded_validation_per_profile": 640,
            "C_factors_9I": [0.01, 0.0316227766, 0.1, 0.3162277660, 1.0],
            "final_C": "C_reference * mean_actual_fold_training_rows / 1055198",
        },
    }
