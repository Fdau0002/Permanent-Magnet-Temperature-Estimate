"""Resolve public original CSV, verify bytes, and print an input-schema audit."""

import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pmsm_project.data import resolve_csv, file_sha256, validate_dataset

p = resolve_csv(ROOT)
frame = pd.read_csv(p)
audit = {"sha256": file_sha256(p), **validate_dataset(frame)}
# Descriptive source ranges are not physical acceptance limits or tuning rules.
ranges = frame.agg(["min", "max"]).T.rename_axis("CSV_field").reset_index()
report_dir = ROOT / "docs/reports"
report_dir.mkdir(parents=True, exist_ok=True)
ranges.to_csv(report_dir / "data_observed_ranges.csv", index=False)
print(json.dumps(audit, indent=2))
