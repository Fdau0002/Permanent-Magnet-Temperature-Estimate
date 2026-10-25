"""Separate lightweight cProfile run of production functions; no model selection."""

from pathlib import Path
import cProfile
import io
import json
import pstats
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def workload(profile):
    """Run real input/features/QR functions on one small development prefix."""
    from pmsm_project.features import make_paper_inputs, make_temporal_features
    from pmsm_project.numerics import compress_profile_ols

    inputs = make_paper_inputs(profile)
    temporal = make_temporal_features(inputs, profile.profile_id, [10, 50, 200, 1000])
    compressed = compress_profile_ols(profile, [10, 50, 200, 1000])
    return temporal.shape, compressed["R"].shape


def main():
    """Profile production code independently of historical scientific timers."""
    import pandas as pd
    from pmsm_project.data import resolve_csv
    from pmsm_project.config import LOCKED_DEVELOPMENT_IDS

    path = resolve_csv(ROOT, allow_download=False)
    # Read a small chunk, then select only an established development recording.
    prefix = pd.read_csv(path, nrows=1000)
    chosen = (
        prefix.loc[prefix.profile_id.isin(LOCKED_DEVELOPMENT_IDS)].iloc[:256].copy()
    )
    if len(chosen) != 256 or chosen.profile_id.nunique() != 1:
        raise ValueError(
            "The first source chunk does not provide one 256-row development prefix"
        )
    profiler = cProfile.Profile()
    tracemalloc.start()
    started = time.perf_counter()
    shapes = profiler.runcall(workload, chosen)
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    destination = ROOT / "docs/reports"
    destination.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(destination / "production_profile.pstats")
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
        "cumulative"
    ).print_stats(35)
    (destination / "production_profile.txt").write_text(stream.getvalue())
    result = {
        "scope": "separate lightweight production-code profile, not historical experiment timing",
        "rows": len(chosen),
        "profile_id": int(chosen.profile_id.iloc[0]),
        "diagnostic_spans": [10, 50, 200, 1000],
        "feature_shape": shapes[0],
        "qr_shape": shapes[1],
        "elapsed_seconds_with_profiling": seconds,
        "python_tracemalloc_peak_bytes": peak,
        "includes_full_dataset_loading": False,
        "native_allocation_peak_claimed": False,
        "production_budgets_changed": False,
    }
    (destination / "production_profile.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
