"""Regenerate source documentation from the actual production package using pdoc."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
subprocess.run(
    [
        sys.executable,
        "-m",
        "pdoc",
        "--docformat",
        "numpy",
        "-o",
        str(ROOT / "docs/html"),
        "pmsm_project",
    ],
    cwd=ROOT,
    env=env,
    check=True,
)
