"""Regenerate chapter/source registers and optional offline notebook HTML previews."""

from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def ast_of(cell):
    """Return syntax-only source identity for the available historical code."""
    source = "".join(cell["source"]).replace("%pip -q install scikit-optimize", "")
    return ast.dump(ast.parse(source), include_attributes=False)


def registers():
    """Write exact available-source matches and experiment-to-test crosswalks."""
    source_root = ROOT / "provenance/source_notebooks"
    fresh = json.loads((source_root / "fresh_reconstruction.ipynb").read_text())
    archive = json.loads(
        (source_root / "recovered_full_data_archive.ipynb").read_text()
    )
    source_trees = {}
    for i, cell in enumerate(archive["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            source_trees[i] = ast_of(cell)
        except SyntaxError:
            pass
    matches = []
    for i in [
        50,
        53,
        56,
        57,
        65,
        66,
        67,
        73,
        74,
        78,
        80,
        86,
        89,
        91,
        93,
        95,
        100,
        102,
        108,
    ]:
        tree = ast_of(fresh["cells"][i])
        hits = [j for j, s in source_trees.items() if tree == s]
        matches.append(
            {
                "fresh_source_cell": i,
                "archive_source_cells": hits,
                "exact_ast_match": bool(hits),
                "ast_sha256": hashlib.sha256(tree.encode()).hexdigest(),
                "note": (
                    "fresh feasibility-seed / portable search setup"
                    if i in (89, 93)
                    else "literal recovered procedure"
                ),
            }
        )
    (ROOT / "provenance/recovered_protocol_matches.json").write_text(
        json.dumps(matches, indent=2)
    )
    sections = json.loads((ROOT / "provenance/educational_sections.json").read_text())
    locations = defaultdict(set)
    for name in ["02_Experimental_Development", "01_Final_Reproducible_Evaluation"]:
        notebook = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
        for cell in notebook["cells"]:
            meta = cell["metadata"]
            if cell["cell_type"] == "code" and meta.get("source_cell") is not None:
                locations[(name[:2], meta.get("educational_section"))].add(
                    f"{meta.get('source_notebook')}:{meta['source_cell']}"
                )
    lines = [
        "# Experiment, research-question and evidence register",
        "",
        "Sections link the actual chronological source computations to numbered explanations. Source indices are zero-based; generated results exist only after execution. All source cells carry metadata and statement-preservation records. Tests verify particular software/numerical properties, not completed full predictive validation.",
        "",
    ]
    for number in ["02", "01"]:
        lines += [
            f"## Notebook {number}",
            "",
            "| Section | Experiment / research question | Original source cells | Generated evidence / verification |",
            "|---|---|---|---|",
        ]
        for row in sections:
            if row["notebook"] != number:
                continue
            sec = row["section"]
            chapter = int(sec.split(".")[0])
            if number == "02":
                tests = (
                    "test_data.py / test_sampling_validation.py"
                    if chapter in (2, 3, 5)
                    else (
                        "test_features.py / test_original_parity.py"
                        if chapter == 4
                        else (
                            "test_handoff.py"
                            if chapter == 13
                            else "test_recovered_protocol.py / test_numerics_models_metrics.py"
                        )
                    )
                )
            else:
                tests = (
                    "test_handoff.py"
                    if chapter == 4
                    else (
                        "test_checkpoints_plotting.py / test_numerics_models_metrics.py"
                        if chapter in (8, 9, 10, 11)
                        else "test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py"
                    )
                )
            refs = (
                "<br>".join(sorted(locations[(number, sec)]))
                or "read-only explanatory/result presentation"
            )
            evidence = output_for(number, sec)
            lines.append(
                f"| {sec} | {row['title']} — {row['research_questions']} | {refs} | {evidence}; {tests} |"
            )
        lines.append("")
    lines += [
        "## Figure trace and captions",
        "",
        "Development figures are exported to the current UUID run’s figures/ directory. figure_captions.json records notebook, run ID, numbered section, original source notebook/cell, titles and axis labels. Captions and interpretation are in that section; raw tables are retained in the corresponding experiment directories. The hook runs outside the original numerical fitting/prediction timers.",
        "",
        "| Final figure | Underlying table or calculation | Meaning / units |",
        "|---|---|---|",
        "| 02_profile_split.png | profile_partition.csv | Session durations (hours), profile IDs and fixed partition |",
        "| 03_ewma_memory.png | original development current and adjusted filters | Current (A), elapsed seconds; diagnostic spans only |",
        "| 04_grouped_cv.png | development_cv_folds.csv | Whole-profile validation-fold membership |",
        "| 08_training_times.png | training_times.csv | Actual seconds under the documented model-dependent fit boundaries |",
        "| 10_accuracy_and_cost.png | overall_metrics.csv + inference_times.csv | RMSE (°C), full inference-loop seconds |",
        "| 11_profile_rmse_bias_heatmaps.png | profile_metrics.csv | Session/model RMSE and signed bias (°C) |",
        "| 11_bias_decomposition.png | profile_bias_diagnostics.csv | Percentage of session MSE from squared mean bias |",
        "| 11_temperature_history.png | original longest held-out profile and saved NPZ predictions | Measured/estimated temperature (°C), elapsed minutes; display-only decimation |",
        "",
        "Final source-row indices and model/manifest hashes trace every metric back to the original measurements. No figure implies execution merely because its generating code is present. Some original developmental figures are diagnostic rather than publication figures; retain their explanatory caption and unit interpretation when using them in the report.",
    ]
    (ROOT / "docs/EXPERIMENT_REGISTER.md").write_text("\n".join(lines) + "\n")


def output_for(number, section):
    """Name the actual evidence family associated with a numbered experiment."""
    chapter = int(section.split(".")[0])
    if number == "01":
        if chapter == 4:
            return "configuration_agreement.csv and imported_development_manifest.json"
        if chapter in (6, 7):
            return "training_manifest.json / training_times.csv"
        if chapter in (8, 9):
            return "84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv"
        if chapter == 10:
            return "per-profile diagnostic CSVs and labelled figures"
        if chapter == 11:
            return "run_manifest.json / reference_comparison.csv / compact results ZIP"
        return "input/fold/feature/QR verification outputs and figures"
    if chapter == 6:
        return "OLS pilot/expanded trial tables and original search plots"
    if chapter in (7, 10):
        return "per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings"
    if chapter == 8:
        return "initial/fixed-validation CSVs; 36 nested timing/error records"
    if chapter == 9:
        return "full_trials / full OLS fold, rank and numerical parity tables"
    if chapter == 11:
        return "original feasibility measurements and scaling summaries"
    if chapter == 12:
        stage = int(section.split(".")[1])
        if stage == 1:
            return "printed fresh RBF spans/C/epsilon and invariant checks"
        if stage == 2:
            return "stage9_data and 108-feature parity checks"
        if stage == 3:
            return (
                "training-only standardisation and matched-kernel reconstruction error"
            )
        if stage in range(4, 12):
            return (
                f"Stage 9{'DEFGHIJK'[stage-4]} original CSV records, warnings and plots"
            )
        if stage == 12:
            return "X9_full / y9_full construction and original ordering checks"
        return "original Stage 9M trained artifact, convergence/persistence manifest and measurements"
    if chapter == 13:
        return "authoritative manifest + run-status JSON + relative evidence CSVs + portable handoff ZIP"
    return "source/schema/split/causal-feature/preliminary verification outputs"


def previews():
    """Render notebooks to optional offline HTML; Pandoc with MathML is required."""
    destination = ROOT / "docs/notebook_previews"
    destination.mkdir(parents=True, exist_ok=True)
    style = """body{font:16px/1.55 system-ui,sans-serif;color:#173047;background:#fff;max-width:1100px;margin:36px auto;padding:0 28px}h1,h2,h3,h4{line-height:1.22;color:#112a3b;scroll-margin-top:22px}h1{font-size:32px;border-bottom:3px solid #2a7791;padding-bottom:20px}h2{margin-top:64px;border-bottom:1px solid #b7ccd4;padding-bottom:10px}h3{margin-top:42px}h4{font-size:17px;color:#235b70}p{max-width:1000px}a{color:#176582}table{border-collapse:collapse;font-size:14px;display:block;overflow-x:auto}td,th{border:1px solid #c4d4dc;text-align:left;padding:9px;vertical-align:top}th{background:#edf4f7}pre{background:#f3f6f8;border:1px solid #d8e1e5;border-radius:5px;padding:16px;overflow:auto;line-height:1.4;font-size:13px}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}math[display=block]{display:block;overflow:auto;padding:14px 0}#TOC{background:#f3f7f9;border:1px solid #d3e1e6;padding:16px 24px;max-height:340px;overflow:auto}blockquote{border-left:3px solid #407d95;padding-left:16px;color:#37556c}"""
    (destination / "style.css").write_text(style)
    for name in ["02_Experimental_Development", "01_Final_Reproducible_Evaluation"]:
        n = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
        parts = []
        for cell in n["cells"]:
            s = "".join(cell["source"])
            parts.append(
                s.replace("../docs/", "../")
                if cell["cell_type"] == "markdown"
                else "\n```python\n" + s + "\n```\n"
            )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "notebook.md"
            source.write_text("\n\n".join(parts))
            subprocess.run(
                [
                    "pandoc",
                    str(source),
                    "--from",
                    "markdown+tex_math_dollars",
                    "--to",
                    "html5",
                    "--standalone",
                    "--mathml",
                    "--toc",
                    "--toc-depth=3",
                    "--css",
                    "style.css",
                    "--metadata",
                    f'title={name.replace("_"," ")}',
                    "-o",
                    str(destination / f"{name}.html"),
                ],
                check=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previews",
        action="store_true",
        help="Also use Pandoc to render offline MathML previews",
    )
    args = parser.parse_args()
    registers()
    if args.previews:
        previews()


if __name__ == "__main__":
    main()
