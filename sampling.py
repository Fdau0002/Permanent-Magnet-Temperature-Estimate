"""Deterministic, nested time-position selection used in Stages 5 and 9.

The source 2 Hz signal is always filtered before rows are sampled. Sampling
changes the number of regression observations, not the filter's timestep.
"""

import heapq
import numpy as np


def nested_sample_order(profile_length: int, max_samples: int) -> list[int]:
    """Select nested observation positions by largest-interval bisection.

    Parameters
    ----------
    profile_length : int
        Number of chronological 2 Hz observations in one profile.
    max_samples : int
        Required count, satisfying 2 <= max_samples <= profile_length.

    Returns
    -------
    list of int
        Unique positions in selection rank order, beginning with both endpoints.
        Sort a prefix only after selecting its desired nested training budget.

    Raises
    ------
    ValueError
        If the requested sample count is outside the supported range.

    Notes
    -----
    Tie-breaking uses heap tuples (-interval_length, left, right). It is fully
    deterministic. Filtering must precede sampling: these positions are not a
    new sample rate for the EWMA. No claim of statistically independent rows
    is implied by selecting them."""
    if not (2 <= max_samples <= profile_length):
        raise ValueError("Require 2 <= max_samples <= profile_length")
    selected = [0, profile_length - 1]
    intervals = [(-(profile_length - 1), 0, profile_length - 1)]
    while len(selected) < max_samples:
        _, left, right = heapq.heappop(intervals)
        midpoint = (left + right) // 2
        if midpoint in (left, right):
            continue
        selected.append(midpoint)
        for a, b in ((left, midpoint), (midpoint, right)):
            if b - a > 1:
                heapq.heappush(intervals, (-(b - a), a, b))
    assert len(selected) == len(set(selected)) == max_samples
    return selected
