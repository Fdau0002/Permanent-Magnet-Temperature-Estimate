"""Execute corrected notebooks in source order using fresh IPython processes.

The full run retains every original budget. --stop-after-source-cell is an
explicit partial source-execution check and cannot create a final handoff.
"""

from pathlib import Path
import argparse, base64, json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def worker(stage, stop_after):
    """Execute actual code cells and checkpoint their genuine outputs."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from matplotlib_inline.backend_inline import configure_inline_support

    name = (
        "02_Experimental_Development"
        if stage == "development"
        else "01_Final_Reproducible_Evaluation"
    )
    notebook = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    if stop_after is not None and not any(
        cell.metadata.get("source_notebook") == "fresh_reconstruction.ipynb"
        and cell.metadata.get("source_cell") == stop_after
        for cell in notebook.cells
    ):
        raise ValueError(
            f"No recovered development source cell {stop_after}; refusing an unbounded partial run"
        )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    destination = Path(os.environ["PMSM_OUTPUT_DIR"]) / "execution_snapshots"
    destination.mkdir(parents=True, exist_ok=True)
    status = {
        "stage": stage,
        "status": "running",
        "source_notebook": name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cells": [],
        "backend": "fresh-process IPython InteractiveShell",
        "scope": "partial" if stop_after is not None else "full",
    }
    shell = InteractiveShell.instance()
    configure_inline_support(shell, "module://matplotlib_inline.backend_inline")
    started = time.perf_counter()

    def save():
        status["elapsed_seconds"] = time.perf_counter() - started
        notebook.metadata["current_execution"] = status
        nbformat.write(notebook, destination / f"{name}_executed.ipynb")
        (destination / f"{name}_status.json").write_text(json.dumps(status, indent=2))

    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if (Path(os.environ["PMSM_OUTPUT_DIR"]) / "STOP_REQUESTED").exists():
                raise KeyboardInterrupt(
                    "User requested cancellation before the next cell."
                )
            tick = time.perf_counter()
            print("START", cell.id, flush=True)
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
                shell.run_cell('import matplotlib.pyplot as _plt; _plt.close("all")')
            cell.execution_count = len(status["cells"]) + 1
            if captured.stdout:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stdout", text=captured.stdout
                    )
                )
            if captured.stderr:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stderr", text=captured.stderr
                    )
                )
            for output in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=output.data, metadata=output.metadata
                    )
                )
            error = result.error_before_exec or result.error_in_exec
            status["cells"].append(
                {
                    "id": cell.id,
                    "source_cell": cell.metadata.get("source_cell"),
                    "source_notebook": cell.metadata.get("source_notebook"),
                    "seconds": time.perf_counter() - tick,
                    "status": "failed" if error else "executed",
                }
            )
            save()
            print(
                "END",
                cell.id,
                round(time.perf_counter() - tick, 2),
                "seconds",
                flush=True,
            )
            if error:
                raise error
            if (
                stop_after is not None
                and cell.metadata.get("source_notebook") == "fresh_reconstruction.ipynb"
                and cell.metadata.get("source_cell") == stop_after
                and cell.metadata.get("code_part", 1)
                == cell.metadata.get("code_parts", 1)
            ):
                status["status"] = "partial_check_completed"
                break
        else:
            status["status"] = "completed"
    except BaseException as exc:
        status["status"] = "failed"
        status["exception"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        print(status["traceback"], flush=True)
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if status["status"] in ["completed", "partial_check_completed"] else 1


def main():
    """Launch development first and final only after a successful handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["development", "final", "both"], default="both"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--stop-after-source-cell", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop_after_source_cell is not None and args.stage != "development":
        parser.error(
            "Partial checks apply only to development; they never continue to final evaluation"
        )
    env = dict(
        os.environ,
        PMSM_PROJECT_ROOT=str(ROOT),
        PMSM_OUTPUT_DIR=str(args.output_root.resolve()),
        PMSM_RUN_MODE="full",
        PYTHONPATH=str(ROOT / "src"),
    )
    for key in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
        env.setdefault(key, "2")
    if args.worker:
        os.environ.update(env)
        os.chdir(ROOT)
        return worker(args.stage, args.stop_after_source_cell)
    for stage in ["development", "final"] if args.stage == "both" else [args.stage]:
        command = [
            sys.executable,
            __file__,
            "--stage",
            stage,
            "--output-root",
            str(args.output_root.resolve()),
            "--worker",
        ]
        if args.stop_after_source_cell is not None:
            command += ["--stop-after-source-cell", str(args.stop_after_source_cell)]
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
