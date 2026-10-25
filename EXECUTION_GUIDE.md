# Execute the original workflow and transfer its selections

## 1. Before Run All

Keep the complete extracted package. The notebooks import `src/pmsm_project`; uploading just an IPYNB to an otherwise empty Colab runtime cannot provide those modules. Preserve the exact original CSV bytes, profile IDs and dependency specification. Run Notebook 02 before Notebook 01.

The full historical budgets are intentional. Do not substitute smoke data, a smaller trial count or older winners because a runtime is slow. If the original setting exceeds the available runtime, preserve the failure/partial status and use a sufficient runtime. This package does not claim a guaranteed completion time or Colab memory fit.

## 2. Prepare Google Colab

First open an empty temporary Colab notebook. Upload the complete delivered ZIP through the Files panel or `files.upload()`. This setup snippet locates its single project root and installs the exact direct dependencies; it performs no numerical experiments:

```python
from google.colab import files
from pathlib import Path
import io, os, sys, zipfile, subprocess

uploaded = files.upload()   # choose MMA3001_Connected_Project.zip
package_name = next(name for name in uploaded if name.endswith('.zip'))
with zipfile.ZipFile(io.BytesIO(uploaded[package_name])) as archive:
    archive.extractall('/content')
project = Path('/content/MMA3001_PMSM_Project')
assert (project/'src/pmsm_project').is_dir()
print('PROJECT ROOT:', project)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(project/'requirements.txt')])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '-e', str(project)])
subprocess.check_call([sys.executable, '-m', 'pip', 'check'])
```

If numerical packages had already been imported, use **Runtime → Restart session** after installation. Restarting the session keeps `/content` files; deleting/disconnecting the runtime may remove them. Do not claim old loaded NumPy/scikit-learn objects use newly installed package versions.

Download/open `notebooks/02_Experimental_Development.ipynb` from the extracted package in Colab. Confirm that runtime contains the extracted project; if it does not, repeat extraction in that notebook’s runtime. Before Run All, execute this small path setup cell in that runtime (or set the two environment variables through an equivalent Colab cell):

```python
from pathlib import Path
import os
project = Path('/content/MMA3001_PMSM_Project')
assert (project/'src/pmsm_project').is_dir()
os.environ['PMSM_PROJECT_ROOT'] = str(project)
os.environ['PMSM_OUTPUT_DIR'] = str(project/'results')
os.chdir(project)
```

This environment setup is outside the scientific protocol and does not change any algorithm, trial count or sample budget. Both production notebooks discover the source tree from the current directory or `PMSM_PROJECT_ROOT`.

## 3. Supply the dataset

The loader attempts public retrieval of `wkirgsn/electric-motor-temperature/versions/3`, original file `measures_v2.csv`, if no local path is supplied. Public access may fail because of network or depositor access restrictions. In that case, manually obtain the original version 3 file from the depositor’s listing or use the supplied original attachment; upload it to `/content` and set:

```python
os.environ['PMSM_CSV_PATH'] = '/content/PMSM Electric Motor Temperature Measurement.csv'
```

Use the actual uploaded filename. Alternatively place it at `project/'data/measures_v2.csv'`. The loader requires SHA-256:

```text
78f3d150f0f2ad9c5dc7ff24dd12c00d386ad530f48c1589ad24fcd88867d3ad
```

Do not re-save, sort or reindex the raw CSV. It has 1,330,816 rows, 13 ordered columns and 69 chronological profile blocks. The byte check deliberately rejects a different version even if its column names look correct.

## 4. Run Notebook 02 completely

Choose **Run all** only after environment/data setup. The notebook creates a new UUID and an active-run status marker. It computes every original pilot, expanded/full OLS campaign, five-model larger campaign, numerical experiment and Stage 9 continuation. Existing prior-run tables/models are not used to supply new optima.

Completion requires all original stages, the Stage 9M convergence gate and strict evidence export. The end displays six model settings and prints:

```text
results/full/development/development_selected_configurations.json
results/full/development/development_run_status.json
results/full/development/development_handoff.zip
```

Evidence CSVs are beneath `results/full/development/runs/<run_id>/handoff_evidence/`. Other current-run tables/figures remain under the same run directory. The handoff ZIP includes the manifest, completion marker and required evidence CSVs. It excludes the 300-MB dataset and large fitted estimators.

Save the **executed notebook** using Colab’s notebook-download command. Also download `development_handoff.zip` through the Files panel, or, after Run All:

```python
from google.colab import files
files.download(str(HANDOFF_ARCHIVE))
```

Save other development tables/figures if needed for the report. A downloadable ZIP does not persist your runtime; keep the downloaded handoff somewhere durable before disconnecting.

## 5. Run Notebook 01 with that exact handoff

If using the same runtime/project/output root, the manifest is found automatically. Ensure Notebook 01 is running in that prepared runtime, set the path environment above if necessary, and Run All.

For a new runtime, repeat package installation/restart and dataset setup. Upload the exact `development_handoff.zip`, then extract it to the expected directory:

```python
from pathlib import Path
import os, zipfile
project = Path(os.environ['PMSM_PROJECT_ROOT'])
handoff_dir = project/'results/full/development'
handoff_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile('/content/development_handoff.zip') as archive:
    archive.extractall(handoff_dir)
os.environ['PMSM_CONFIG_MANIFEST'] = str(handoff_dir/'development_selected_configurations.json')
```

Open Notebook 01 in that runtime and Run All. Its loader checks all six settings, evidence hashes, dataset/source hashes, original search protocol, fold map, dependency versions and completion marker **before any fit**. The displayed agreement table must have six True rows. No old parameters are tried if import fails.

Final training retains the established sequence OLS, k-NN, Extra Trees, MLP, Random Forest, LinearSVR. Every model uses all 1,055,198 development rows in full mode. Only then are 14 held-out profiles predicted, producing 84 source-aligned checkpoints. A stopping/convergence failure remains a real limitation of the selected original configuration; do not silently adjust it using test error.

## 6. Final outputs and export

Outputs are under `results/full/final/`: `models/` (large joblib artifacts), `predictions/` (84 NPZ files), `results/` (metrics, timings, agreement/imported manifests and audit), and `figures/`. A compact `PMSM_full_results_and_figures.zip` is generated after the consistency audit. It does not contain all large models/predictions. Save those separately if your replication record requires them.

Save the executed Notebook 01. Interpret the freshly calculated tables; historical reference RMSEs are explicitly labelled comparison context. Do not force agreement by changing a parameter. Different fresh winners can produce legitimate different final results.

## 7. Lightweight checks and execution status

```bash
python scripts/verify_reconstruction.py
python -m pytest --html=docs/reports/pytest.html --self-contained-html --junitxml=docs/reports/pytest.xml
python scripts/build_docs.py
```

These verify source consistency, dependencies, synthetic numerical identities, production interfaces and handoff rejection/transfer behaviour. They cannot certify completed expensive optimisation or final predictive accuracy. The separate targeted preprocessing command is:

```bash
python scripts/run_notebooks.py --stage development --stop-after-source-cell 36 --output-root results/preprocessing_check
```

That partial execution stops before Bayesian/model experiments and cannot publish a complete manifest. Production trial budgets remain unchanged. Final notebook smoke mode is available only after a legitimate full development manifest exists; set `PMSM_RUN_MODE=smoke` and use its separately labelled outputs only for workflow testing. Never report smoke metrics as full scientific findings.

## 8. Diagnosing an import failure

| Failure | Resolution preserving the procedure |
|---|---|
| Missing manifest/status/evidence | Transfer the complete handoff ZIP from the successful Notebook 02 run |
| Active/newer development run | Complete that run; do not reuse a superseded selection |
| Dataset mismatch | Restore exact original version 3 bytes |
| Software mismatch | Install the recorded versions and restart loaded runtimes |
| Source mismatch | Use the same package code; if code intentionally changes, regenerate development selection |
| Incomplete/smoke evidence | Finish the original full development protocol |
| Convergence failure | Preserve its record and investigate; no automatic parameter/budget change is applied |

Strict compatibility compares Python major/minor and core numerical versions, not CPU model. Identical software does not ensure identical stochastic optima, floating-point summation order, runtime or memory measurements.
