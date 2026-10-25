# PMSM permanent-magnet temperature estimation — MMA3001

This project estimates permanent-magnet temperature (°C) from observable electrical, mechanical and thermal measurements. It reconstructs the supplied development procedures and connects their **fresh selections** to final evaluation. It uses the public Paderborn PMSM dataset and Kirchgässner, Wallscheid and Böcker (2021); see [references](REFERENCES.md).

**Run `02_Experimental_Development.ipynb` before `01_Final_Reproducible_Evaluation.ipynb`.** Both are under `notebooks/`. Notebook numbering reflects the requested deliverable names, not execution order.

## Current execution status

Both notebooks were completed in Google Colab on 25 October 2026, running Notebook 02 before Notebook 01. Notebook 02 generated fresh model selections and the development handoff. Notebook 01 imported that manifest, verified agreement for all six configurations, trained on the 55 development profiles and evaluated the 14 held-out profiles. Its saved final audit reports 84 model/profile prediction checkpoints.

The subsequent Colab test run completed with 124 passed, 0 failed and 1 warning. The pdoc HTML documentation was regenerated and its structural/link verification passed.

Portable source notebooks remain under `notebooks/`. Completed notebooks with saved outputs are retained under `docs/reports/executed/`, and the recorded handoff is preserved as `docs/reports/development_handoff.zip`. The earlier authoring verification and subsequent student execution are distinguished in `docs/VERIFICATION_REPORT.md`.

## Quick start locally

The completed Colab execution used Python 3.13.16. Reusing its recorded handoff requires Python 3.13 and the exact core dependency versions recorded in the manifest. The earlier authoring checks used Python 3.12.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python -m pip check
```

On Windows activate with `.venv\Scripts\activate` instead. `requirements.txt` pins the directly used packages; `requirements.lock` records the validated full installed environment. For exact transitive pins, install `requirements.lock` before the editable package. The manifest strictly compares the core numerical versions and Python major/minor.

Place the unmodified original `measures_v2.csv` in `data/`, or set `PMSM_CSV_PATH` to the supplied CSV path. Renaming the original file is fine; re-saving its contents changes the hash. When no local file exists, acquisition attempts the depositor’s public Kaggle version 3:

```bash
python scripts/fetch_data.py
python -m pytest --html=docs/reports/pytest.html --self-contained-html --junitxml=docs/reports/pytest.xml
python scripts/verify_reconstruction.py
python scripts/build_docs.py
```

Open the two numbered notebooks in Jupyter/VS Code and Run All in that order. Alternatively, the following commands execute their actual cells in separate fresh IPython processes and save executed snapshots/status files:

```bash
python scripts/run_notebooks.py --stage development
python scripts/run_notebooks.py --stage final
```

Those are **full, potentially expensive runs**. They are provided for independent execution; the authoring verification did not certify a full connected run. The runner saves real outputs and failure state. It has no notebook algorithm substitutes. To request cancellation between cells, create `results/STOP_REQUESTED`; remove it before intentionally starting again. It cannot interrupt a currently executing numerical cell.

## Google Colab

Follow the complete [Colab execution and handoff guide](docs/EXECUTION_GUIDE.md). Upload/extract the complete project ZIP, install dependencies, restart after changing loaded numerical packages, then run Notebook 02. Download its generated `development_handoff.zip`, which contains the JSON selection, completion marker and matching evidence CSVs. Notebook 01 requires all of those files. Transferring JSON alone is insufficient.

If both notebooks use the same prepared runtime and output root, import is automatic. In a new runtime, extract the handoff into `results/full/development/` before Notebook 01. There is no historical fallback; an interrupted new development attempt blocks an old manifest. Missing, altered, smoke, wrong-dataset, wrong-source and wrong-version manifests stop before training.

The package also contains readable [offline notebook HTML previews](docs/notebook_previews/02_Experimental_Development.html). Optional regeneration uses `python scripts/build_review_materials.py --previews` with Pandoc installed; it does not execute notebook experiments. `python scripts/verify_docs.py` checks the generated API/previews and local navigation.

## Original experimental contract

| Component | Preserved procedure |
|---|---|
| Data | 1,330,816 rows, 69 profiles, 2 Hz; exact SHA-256 verification |
| Split | 55 development profiles / 1,055,198 rows; 14 held-out / 275,618 rows; fixed IDs |
| Features | 12 instantaneous inputs plus four causal adjusted EWMA/EWMS spans per input: 108 features |
| Validation | Original profile-stratified grouped three folds; mean fold MSE (°C²) |
| OLS | 8-call GP pilot; 20 additional proposals; 20-evaluation full-data QR GP search including a freshly scored seed |
| Five other Bayesian families | Original GP/EI ask/tell, 8 pilot and 12 larger evaluations each |
| Larger sampling | SVR 320 fitting observations/profile; other families 640; validation 640 |
| Exact-SVR scaling | Nested 80/160/320/640 fitting samples/profile, three folds, three timing repeats |
| Efficient SVR | Original Stage 9A–9M, matched kernels, Nyström/direct-linear comparisons, C/convergence/sample-size sensitivity |
| Final six estimators | OLS, k-NN, Extra Trees, MLP, Random Forest and direct LinearSVR, fitted from the imported manifest |

See [methodology preservation](docs/METHODOLOGY_PRESERVATION.md) for the exact spaces, objectives, source-cell references and necessary infrastructure changes. Historical winners exist only under `provenance/` or in the explicitly labelled retrospective final comparison.

## Package contents

| Path | Purpose |
|---|---|
| `Final Notebooks/` | Numbered, chapterised development and final notebooks |
| `src/pmsm_project/` | Actual imported data, feature, objective, model, QR, metric and handoff code |
| `tests/` | Production-function numerical, interface, provenance and source-preservation tests |
| `docs/html/index.html` | Generated pdoc documentation; open locally in a browser |
| `docs/reports/pytest.html`, `pytest.xml` | Actual pytest HTML/JUnit reports |
| `docs/reports/coverage.xml` | Executed statement-coverage measurement, not scientific coverage |
| `docs/EXPERIMENT_REGISTER.md` | RQ/chapter/source/output/test traceability |
| `docs/COMPLIANCE_CHECKLIST.md` | Brief/rubric requirement status and remaining student work |
| `provenance/source_notebooks/` | Unmodified supplied/recovered notebook sources and saved historical outputs |
| `provenance/chapterisation_code_audit.json` | AST preservation records for partitioned source cells |
| `data/`, `results/` | Locations for retrieved data and regenerated outputs; large files excluded |

## Interpretation and submission boundaries

The final set has already been inspected in earlier project iterations; it is a fixed retrospective evaluation partition, not a pristine project-level confirmation. Development code excludes held-out profiles from selection. Final MLP internal early stopping remains row-based; LinearSVR C×N adjustment remains heuristic; timing boundaries differ between some methods. The workflow is an offline laboratory investigation, not a certified motor-protection system.

The [AI-use record](AI_USE.md) discloses substantial AI assistance and leaves the student’s personal reflection for their review. The separate 5–10-page report, presentation, repository publication/link and meaningful commit history are **not certified complete**. No Git operations were performed. Integrate using [COMMIT_GUIDE.md](COMMIT_GUIDE.md); existing repository files require manual reconciliation.

Source licensing is explained in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The dataset and benchmark PDF are not redistributed in this package.
