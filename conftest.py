"""Independent synthetic measurements and optional original source fixtures."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pmsm_project.config import EXPECTED_COLUMNS


@pytest.fixture
def profile():
    rng = np.random.default_rng(1481)
    n = 360
    t = np.arange(n) / 2
    f = pd.DataFrame(
        {
            "u_q": 40 + 10 * np.sin(t / 7),
            "coolant": 24 + t / 400,
            "stator_winding": 30 + t / 80,
            "u_d": rng.normal(0, 4, n),
            "stator_tooth": 29 + t / 90,
            "motor_speed": 1000 + 400 * np.cos(t / 15),
            "i_d": rng.normal(0, 8, n),
            "i_q": 20 + 5 * np.sin(t / 8),
            "pm": 28 + 0.04 * t + np.sin(t / 30),
            "stator_yoke": 27 + t / 100,
            "ambient": 22 + t / 1000,
            "torque": 8 + 2 * np.cos(t / 10),
            "profile_id": np.repeat(2, n),
        }
    )
    return f[EXPECTED_COLUMNS]


@pytest.fixture(scope="session")
def original_csv():
    from pmsm_project.data import resolve_csv

    root = Path(__file__).resolve().parents[1]
    try:
        return resolve_csv(root, allow_download=False)
    except FileNotFoundError:
        pytest.skip(
            "Original CSV unavailable: set PMSM_CSV_PATH for integration checks"
        )
