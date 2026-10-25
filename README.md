# Permanent-Magnet Temperature Estimation | MMA3001

**Mechanical engineering numerical methods and machine learning project**  
**Goal:** Estimate the permanent-magnet temperature of a permanent-magnet synchronous motor (PMSM), in °C, from electrical, mechanical, and thermal operating measurements.

> **For teaching assistants — start here.** This repository contains (1) clean, executable source notebooks, (2) separate completed Google Colab notebooks with visible outputs, (3) a saved *development-to-final* configuration handoff, and (4) the Python implementation, tests, documentation and selected development evidence. **The original ~300 MB dataset and large fitted final models are not included in the repository.** The completed notebooks can be reviewed without rerunning the experiments.

## 1. Fastest way to review the submission

| What you want to check | Open this | What it proves / does not prove |
|---|---|---|
| Read the **completed development work** | [`Final Notebooks/02_Experimental_Development (Final).ipynb`](Final%20Notebooks/02_Experimental_Development%20%28Final%29.ipynb) | Saved outputs for model comparisons, optimisation, scaling and the exported handoff. This is **execution evidence**, not a portable fresh-run entry point. |
| Read the **completed final evaluation** | [`Final Notebooks/01_Final_Reproducible_Evaluation (Final).ipynb`](Final%20Notebooks/01_Final_Reproducible_Evaluation%20%28Final%29.ipynb) | Saved outputs from final model fitting, configuration checks, per-profile predictions and comparison metrics. Its added Colab setup cell refers to a past runtime path. |
| See the saved development selections | [`results/full/development/development_selected_configurations.json`](results/full/development/development_selected_configurations.json) | The six chosen model configurations, run ID, split, software versions, source digest and evidence-file hashes. |
| Inspect the transferable selection archive | [`results/full/development/development_handoff.zip`](results/full/development/development_handoff.zip) | Manifest + completion marker + 19 hashed evidence CSVs for Notebook 01. No dataset or trained final models. |
| Inspect clean **runnable** notebooks | [`notebooks/02_Experimental_Development.ipynb`](notebooks/02_Experimental_Development.ipynb), then [`notebooks/01_Final_Reproducible_Evaluation.ipynb`](notebooks/01_Final_Reproducible_Evaluation.ipynb) | Original source code with outputs cleared, intended for independent execution. **02 runs before 01**, despite the names. |
| Inspect tests and prior execution reports | [`tests/`](tests/), [`docs/reports/pytest.html`](docs/reports/pytest.html), [`docs/VERIFICATION_REPORT.md`](docs/VERIFICATION_REPORT.md) | Recorded automated checks, with separate discussion of the scientific full runs. |
| Read the method and code | [`docs/METHODOLOGY_PRESERVATION.md`](docs/METHODOLOGY_PRESERVATION.md), [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md), [`src/pmsm_project/`](src/pmsm_project/) | Exact experimental contract, input units, computations and interface documentation. |

**Execution status.** On 25 October 2026, the saved Colab notebooks show a completed Notebook 02 development run, a completed Notebook 01 final run, six imported model configurations and **84 model/profile prediction checkpoints** (6 estimators × 14 held-out profiles). The accompanying recorded automated suite reports **124 passed, 0 failed**, with one warning. These are **saved execution records**, not a claim that every TA machine will reproduce identical results or timings.

## 2. What the software does

The dataset is the publicly available *Electric Motor Temperature* dataset (Paderborn University / Kirchgässner and colleagues), **Kaggle dataset version 3, `measures_v2.csv`**. It consists of **1,330,816 observations**, sampled at a documented **2 Hz**, across **69 recording profiles**. The prediction target is `pm` (permanent-magnet temperature, °C). The target, profile identifier, winding/tooth/yoke temperatures and measured torque are **not model predictors** in the chosen configuration.

- **Inputs:** Seven measured operating variables (`ambient`, `coolant`, `u_d`, `u_q`, `i_d`, `i_q`, `motor_speed`) plus five derived quantities (`u_s`, `i_s`, `S_el`, `i_s_omega`, `S_el_omega`) = **12 instantaneous predictor channels**.
- **History:** For each channel, the code adds four causal exponentially weighted means (EWMA) and four exponentially weighted standard deviations (EWMS), with profile-boundary resets. That gives **12 + 12 × 8 = 108 features**, where four history spans are tuned per model.
- **Data separation:** **55 development profiles / 1,055,198 rows** for selection and fitting, and **14 held-out profiles / 275,618 rows** for final evaluation. Selection uses three **profile-disjoint**, stratified validation folds. The 14-profile test set is not used by the code to choose new parameters.
- **Models:** Ordinary least squares (OLS), k-nearest neighbours (k-NN), Extra Trees, multilayer perceptron (MLP), Random Forest and direct LinearSVR. The development notebook also investigates exact RBF-SVR and efficient SVR variants.
- **Evaluation:** Grouped validation MSE (°C²) during development; final RMSE/MAE, bias, R², timings, per-profile comparisons and row-aligned prediction evidence on the fixed held-out group.

The central experimental questions concern **history and feature construction, model/parameter selection, SVR scalability, and numerical validity/sensitivity**. See [`docs/EXPERIMENT_REGISTER.md`](docs/EXPERIMENT_REGISTER.md) for the chapter-by-chapter trace from original computations to evidence.

### Execution and data flow

```text
Original Kaggle CSV (not included; SHA-256 verified)
                    |
                    v
notebooks/02_Experimental_Development.ipynb
  - validate data, allocate profiles, build causal features
  - run grouped-CV searches and numerical/scaling experiments
  - export six freshly selected configurations + 19 evidence CSVs
                    |
                    v
results/full/development/development_handoff.zip
  - development_selected_configurations.json
  - development_run_status.json
  - runs/<run_id>/handoff_evidence/*.csv
                    |
                    v
notebooks/01_Final_Reproducible_Evaluation.ipynb
  - reject incompatible/incomplete handoffs before fitting
  - fit six final models on 55 development profiles
  - evaluate each on 14 held-out profiles (84 checkpoints)
                    |
                    v
results/full/final/  (generated; not bundled here)
  - results/ metrics, audit and comparison tables
  - figures/ diagnostic plots
  - models/ and predictions/ large artifacts
```

**Why the numbering is reversed:** `02` is the exploratory/development notebook and must finish before `01`, which loads its configuration evidence for final training and evaluation. Do not replace model configurations with historical values to bypass this contract.

## 3. Choose a review or reproduction route

**A. Review saved results (no dataset or computing setup needed).** Open the two notebooks in `Final Notebooks/`, with Notebook 02 first. Those are *records of completed runs*, not the notebooks intended for a fresh Run All. They include Colab-specific setup/download cells whose `/content/...` locations refer to the original runtime. Inspect model selections, the six-row configuration-agreement output and final metric tables in the outputs.

**B. Re-run Notebook 01 using the supplied completed handoff (faster than rerunning model search).** Use the clean notebook under `notebooks/`, the original CSV and **Python 3.13** with the pinned core packages. The saved manifest/status/evidence already exist under `results/full/development/`; Notebook 01 reads them by default. This still performs **six full-data fits and held-out evaluation**, and can be computationally demanding. The source fingerprint, raw CSV hash and software versions must match; do not edit the manifest to force acceptance.

**C. Re-run everything from scratch.** Use an independent copy of the repository, set up the original CSV/environment, run **Notebook 02 and then Notebook 01**. A fresh Notebook 02 run creates a new active-run marker and subsequently a new handoff. **Do not interrupt Notebook 02 halfway and then expect the old saved handoff to remain usable** in that same output directory. New selections, timings and metrics may legitimately differ. The original trial budgets are not reduced for a quick demonstration.

## 4. Local setup and commands

From the extracted **repository root** (the directory containing `pyproject.toml`, `notebooks/` and `src/`), use Python **3.13.x** when you intend to load the included saved handoff. Its software record specifically requires Python major/minor `3.13` and exact pinned numerical-library versions. The earlier authoring verification used Python 3.12; **3.12 cannot import the saved 3.13 handoff** without rerunning development under 3.12.

```bash
# Linux/macOS
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python -m pip check
```

On Windows use `py -3.13 -m venv .venv` and `.venv\Scripts\activate`, then the same installation commands. `requirements.txt` pins direct packages; `requirements.lock` is the recorded complete environment if exact transitive pins are needed.

**Supply the dataset.** This ZIP deliberately excludes the 300,061,411-byte CSV. Either put the original `measures_v2.csv` in a new `data/` directory at the repository root, or set `PMSM_CSV_PATH` to the original file's absolute path. Without a local file, the loader attempts to retrieve `wkirgsn/electric-motor-temperature/versions/3` through KaggleHub; that requires network access and may fail. No resaving, sorting or editing of the CSV is allowed. Expected SHA-256:

```text
78f3d150f0f2ad9c5dc7ff24dd12c00d386ad530f48c1589ad24fcd88867d3ad
```

An optional data integrity command (will attempt the download if the file is not present):

```bash
python scripts/fetch_data.py
```

**Lightweight verification** (run after installation; original-data integration tests require the CSV):

```bash
python -m pytest -q
python scripts/verify_reconstruction.py
python scripts/verify_docs.py
```

These tests are not substitutes for the expensive optimisation and final-fit experiments. For a **full fresh run**, use the clean notebooks in Jupyter/VS Code, or use the supplied notebook runner (which saves checkpoints in `results/execution_snapshots/`):

```bash
python scripts/run_notebooks.py --stage development
python scripts/run_notebooks.py --stage final
# Alternatively: python scripts/run_notebooks.py --stage both
```

For **saved-handoff final evaluation only**, skip the development command, keep the existing `results/full/development/` contents, and run `--stage final` after setting up the matching environment and dataset. The final command creates large results and may take substantial time/memory. Start from a separate directory/copy if you want to preserve the bundled records.

### Google Colab

The clean notebooks can also run in Colab, but each runtime must have the **whole repository**, not just an uploaded `.ipynb`. Upload/extract this repository ZIP, find the directory containing `src/pmsm_project`, install `requirements.txt`, and set `PMSM_PROJECT_ROOT` to that extracted directory (and optionally `PMSM_OUTPUT_DIR` to its `results/` folder). **The extracted directory for a GitHub ZIP is `Permanent-Magnet-Temperature-Estimate-main/`, not `MMA3001_PMSM_Project/`.** Restart the runtime after altering loaded numerical packages, and re-establish environment variables in each new runtime. Check that the runtime's Python is 3.13 before importing the saved handoff.

The Colab process, file transfer, and handoff procedure are detailed in [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md). Remember that the added setup cells in `Final Notebooks/` are evidence of the original Colab runs, not portable installation instructions.

## 5. Where the files belong

| Directory/file | Role |
|---|---|
| `notebooks/` | **Canonical clean source** of the two runnable notebooks. Source code from both is part of the development-manifest fingerprint. |
| `Final Notebooks/` | **Saved execution snapshots** manually retained from Colab, with printed results, figures and extra Colab cells. Review these; do not use them as the fingerprinted fresh-run source. |
| `src/pmsm_project/` | Production modules for strict data loading, causal features, sampling, search, estimators, numerical methods, metrics, checkpointing and fail-closed handoff verification. |
| `results/full/development/` | Bundled completed development manifest, run-status marker, 19 evidence files, the handoff ZIP, and selected development runs/figures. |
| `results/full/final/` | Generated by a final run. **Not present in this uploaded repository**; the completed final metrics can be inspected in the saved Notebook 01 outputs. |
| `tests/` | Tests of actual imported functions, provenance, numerical identities, data checks, handoff export/import/rejection and notebook structure. |
| `scripts/` | Dataset preparation, full/partial notebook runner, source verification, report/doc generation and profiling. |
| `docs/html/` | Offline API documentation, beginning at [`docs/html/index.html`](docs/html/index.html). |
| `docs/notebook_previews/` | Read-only HTML versions of the source notebooks; they are **not** the completed notebook-output snapshots. |
| `docs/reports/` | Recorded pytest, coverage, verification and performance reports; reports describe checks performed at different stages. |
| `docs/*.md` | Engineering/data contract, experimental traceability, methodology, execution instructions, performance, verification and report guidance. |
| `provenance/` | Historical source notebooks and research reconstruction/AST audit files; for verification, **not** a source of fallback optimised settings. |
| `AI_USE.md`, `REFERENCES.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE` | AI-use disclosure, references, third-party dataset/licensing information and code license. |

## 6. How the handoff prevents inconsistent evaluation

Notebook 02's exported manifest records its unique run ID, complete status, dataset hash, exact split, fold assignment, search protocol, causal feature contract, installed numerical versions, a digest of the **canonical** `notebooks/` code plus `src/` Python code, six freshly selected configurations, and hashes of the 19 required evidence tables. The companion run-status file binds the completed run to that manifest. Notebook 01 checks these before final fitting; a missing/edited CSV, changed notebook source, newer interrupted development run, incomplete evidence or incompatible Python/library version causes an explicit rejection rather than silently using archived parameters.

When transferring between fresh Colab runtimes, move the **whole** `development_handoff.zip` and extract its contents into `results/full/development/` in the destination project. Transferring only the selection JSON is insufficient. An existing source-compatible development handoff may be reused for Notebook 01; generating new selections requires executing Notebook 02 again.

## 7. Limits and interpretation

- The 14 held-out profiles were not used by the current implementation's development optimiser; however, earlier project iterations inspected results from that fixed partition. This is a **retrospective holdout**, not an entirely pristine project-level test.
- The input dataset is from one documented motor experiment; the models are not certified for another motor, out-of-distribution operation or safety-critical thermal protection. The recorded 2 Hz interval has no explicit per-row timestamps to verify timing jitter.
- Training and inference timings depend on hardware and execution environment. Model-search optima and metrics can vary even when the interface and validated numerical method are unchanged.
- The tests and generated HTML document **code/interface consistency**, not universal predictive accuracy. Large final joblib/NPZ files and the raw CSV are not shipped; reproducing them requires the full-data run.
- The README and notebooks do not replace the separately assessed student report, presentation, personal AI-use reflection, or repository history. See [`AI_USE.md`](AI_USE.md), [`docs/REPORT_GUIDE.md`](docs/REPORT_GUIDE.md), and [`docs/COMPLIANCE_CHECKLIST.md`](docs/COMPLIANCE_CHECKLIST.md).

## 8. Common problems

| Symptom | Cause / remedy |
|---|---|
| `FileNotFoundError` for `notebooks/01_Final_Reproducible_Evaluation.ipynb` | Ensure the clean file has **exactly** that name. `(1)` suffixes and doubled `.ipynb` names are invalid for the runner. |
| `Dataset bytes changed` | Obtain the exact version 3 `measures_v2.csv`; renaming is okay, re-saving is not. |
| `Numerical software versions differ` | Use Python 3.13 and the pinned versions, restart the runtime and rerun the loader. |
| `Production source differs from the development run` | Keep canonical notebook code and `src/` unchanged. The completed Colab snapshots have extra cells and **must not** replace canonical sources. |
| `Development run is incomplete or superseded` | An interrupted new Notebook 02 attempt changed the active marker. Finish that run or recover the corresponding completed handoff in a clean output directory. |
| Manifest or evidence is missing | Restore/extract the complete handoff ZIP with its status JSON and 19 hashed CSVs. |
| Kernel cannot find `pmsm_project` | Open/run from the extracted repository root, install the editable package and set `PMSM_PROJECT_ROOT` if needed. |
| No final `.joblib`/`.npz` files in this repository | Expected: large final products were not committed. Inspect completed Notebook 01 outputs or regenerate final artifacts. |

**Academic provenance:** Source data and principal reference are credited in [`REFERENCES.md`](REFERENCES.md). The code and experiment explanations are educational/research artifacts, not calibrated industrial condition-monitoring software.
