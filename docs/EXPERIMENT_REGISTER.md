# Experiment, research-question and evidence register

Sections link the actual chronological source computations to numbered explanations. Source indices are zero-based; generated results exist only after execution. All source cells carry metadata and statement-preservation records. Tests verify particular software/numerical properties, not completed full predictive validation.

## Notebook 02

| Section | Experiment / research question | Original source cells | Generated evidence / verification |
|---|---|---|---|
| 2.1 | Portable runtime and reproducibility controls — RQ1–RQ4 | fresh_reconstruction.ipynb:4 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 2.2 | Dataset provenance, schema and byte integrity — RQ1 | fresh_reconstruction.ipynb:6<br>fresh_reconstruction.ipynb:7 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 3.1 | Profile statistics and data-quality checks — RQ1 | fresh_reconstruction.ipynb:10<br>fresh_reconstruction.ipynb:12 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 3.2 | Row-split diagnostic and profile-disjoint assignment — RQ1 | fresh_reconstruction.ipynb:15<br>fresh_reconstruction.ipynb:17<br>fresh_reconstruction.ipynb:18 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 4.1 | Measured and derived operating-state inputs — RQ1 | fresh_reconstruction.ipynb:23 | source/schema/split/causal-feature/preliminary verification outputs; test_features.py / test_original_parity.py |
| 4.2 | Causal EWMA/EWMS implementation — RQ1 | fresh_reconstruction.ipynb:25<br>fresh_reconstruction.ipynb:26 | source/schema/split/causal-feature/preliminary verification outputs; test_features.py / test_original_parity.py |
| 4.3 | Numerical and visual verification of temporal memory — RQ1, RQ4 | fresh_reconstruction.ipynb:28<br>fresh_reconstruction.ipynb:30 | source/schema/split/causal-feature/preliminary verification outputs; test_features.py / test_original_parity.py |
| 5.1 | Stratified grouped three-fold validation — RQ1, RQ4 | fresh_reconstruction.ipynb:33<br>fresh_reconstruction.ipynb:36 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 5.2 | Preliminary estimator interface checks — RQ1, RQ2, RQ4 | fresh_reconstruction.ipynb:38<br>fresh_reconstruction.ipynb:40 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 5.3 | Preliminary OLS error localisation — RQ2, RQ4 | fresh_reconstruction.ipynb:42 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 5.4 | Recording-start sensitivity — RQ1, RQ4 | fresh_reconstruction.ipynb:44 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 5.5 | OLS outlier influence and treatment decision — RQ2, RQ4 | fresh_reconstruction.ipynb:46 | source/schema/split/causal-feature/preliminary verification outputs; test_data.py / test_sampling_validation.py |
| 6.1 | Initial OLS Gaussian-process pilot — RQ2, RQ4 | fresh_reconstruction.ipynb:50 | OLS pilot/expanded trial tables and original search plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 6.2 | Review of pilot intervals and boundary evidence — RQ2, RQ4 | fresh_reconstruction.ipynb:52<br>fresh_reconstruction.ipynb:53<br>fresh_reconstruction.ipynb:54 | OLS pilot/expanded trial tables and original search plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 6.3 | Expanded OLS ask/tell search — RQ2, RQ4 | fresh_reconstruction.ipynb:56<br>fresh_reconstruction.ipynb:57<br>fresh_reconstruction.ipynb:58 | OLS pilot/expanded trial tables and original search plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 7.1 | SVR — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 7.2 | k-NN — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 7.3 | Random Forest — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 7.4 | Extra Trees — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 7.5 | MLP — RQ2, RQ3, RQ4 | fresh_reconstruction.ipynb:65<br>fresh_reconstruction.ipynb:66 | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 7.6 | Common five-model Bayesian implementation and pilot comparison — RQ2, RQ3, RQ4 | fresh_reconstruction.ipynb:67<br>fresh_reconstruction.ipynb:68<br>fresh_reconstruction.ipynb:70 | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 8.1 | Initial and fixed-validation RBF-SVR scaling — RQ3, RQ4 | fresh_reconstruction.ipynb:73<br>fresh_reconstruction.ipynb:74<br>fresh_reconstruction.ipynb:75<br>fresh_reconstruction.ipynb:76 | initial/fixed-validation CSVs; 36 nested timing/error records; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 8.2 | Nested sampling and repeated three-fold timings — RQ3, RQ4 | fresh_reconstruction.ipynb:78<br>fresh_reconstruction.ipynb:80<br>fresh_reconstruction.ipynb:83<br>fresh_reconstruction.ipynb:84 | initial/fixed-validation CSVs; 36 nested timing/error records; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 9.1 | Full-development QR OLS feasibility — RQ2, RQ4 | fresh_reconstruction.ipynb:86<br>fresh_reconstruction.ipynb:88<br>fresh_reconstruction.ipynb:89 | full_trials / full OLS fold, rank and numerical parity tables; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 9.2 | QR equivalence and numerical rank verification — RQ4 | fresh_reconstruction.ipynb:91 | full_trials / full OLS fold, rank and numerical parity tables; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 9.3 | Twenty-evaluation full-data Bayesian OLS search — RQ2, RQ4 | fresh_reconstruction.ipynb:93<br>fresh_reconstruction.ipynb:95<br>fresh_reconstruction.ipynb:98 | full_trials / full OLS fold, rank and numerical parity tables; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 10.1 | SVR — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 10.2 | k-NN — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 10.3 | Random Forest — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 10.4 | Extra Trees — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 10.5 | MLP — RQ2, RQ3, RQ4 | fresh_reconstruction.ipynb:100 | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 10.6 | Common larger-search execution and consolidation — RQ2, RQ3, RQ4 | fresh_reconstruction.ipynb:102<br>fresh_reconstruction.ipynb:106 | per-family pilot/larger CSVs, parameter JSON, three fold MSE values and timing/warnings; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 11.1 | Historical training-feasibility measurements — RQ2, RQ3, RQ4 | fresh_reconstruction.ipynb:108<br>fresh_reconstruction.ipynb:111 | original feasibility measurements and scaling summaries; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.1 | Stage 9A: fresh SVR reference and invariants — RQ3, RQ4 | fresh_reconstruction.ipynb:113<br>fresh_reconstruction.ipynb:123 | printed fresh RBF spans/C/epsilon and invariant checks; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.2 | Stage 9B: common screened dataset and feature parity — RQ1, RQ3, RQ4 | fresh_reconstruction.ipynb:125<br>fresh_reconstruction.ipynb:126 | stage9_data and 108-feature parity checks; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.3 | Stage 9C: training-only scaling and Nyström kernel check — RQ1, RQ3, RQ4 | fresh_reconstruction.ipynb:128<br>fresh_reconstruction.ipynb:129 | training-only standardisation and matched-kernel reconstruction error; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.4 | Stage 9D: controlled four-method screening — RQ3, RQ4 | fresh_reconstruction.ipynb:132<br>fresh_reconstruction.ipynb:133 | Stage 9D original CSV records, warnings and plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.5 | Stage 9E: regularisation and convergence sensitivity — RQ3, RQ4 | fresh_reconstruction.ipynb:135<br>fresh_reconstruction.ipynb:136 | Stage 9E original CSV records, warnings and plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.6 | Stage 9F: Nyström approximation capacity — RQ3, RQ4 | fresh_reconstruction.ipynb:138 | Stage 9F original CSV records, warnings and plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.7 | Stage 9G: confirmation on the original outer fold — RQ3, RQ4 | fresh_reconstruction.ipynb:140<br>fresh_reconstruction.ipynb:141 | Stage 9G original CSV records, warnings and plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.8 | Stage 9H: three-fold grouped efficient-SVR comparison — RQ1, RQ3, RQ4 | fresh_reconstruction.ipynb:143<br>fresh_reconstruction.ipynb:144<br>fresh_reconstruction.ipynb:147 | Stage 9H original CSV records, warnings and plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.9 | Stage 9I: direct LinearSVR C grid — RQ3, RQ4 | fresh_reconstruction.ipynb:149<br>fresh_reconstruction.ipynb:150<br>fresh_reconstruction.ipynb:153 | Stage 9I original CSV records, warnings and plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.10 | Stage 9J: larger training samples — RQ3, RQ4 | fresh_reconstruction.ipynb:155<br>fresh_reconstruction.ipynb:156<br>fresh_reconstruction.ipynb:157 | Stage 9J original CSV records, warnings and plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.11 | Stage 9K: sample-size-aware regularisation — RQ3, RQ4 | fresh_reconstruction.ipynb:159<br>fresh_reconstruction.ipynb:160<br>fresh_reconstruction.ipynb:163 | Stage 9K original CSV records, warnings and plots; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.12 | Stage 9L: full-development feature reconstruction — RQ1, RQ3, RQ4 | stage9_original.ipynb:127 | X9_full / y9_full construction and original ordering checks; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 12.13 | Stage 9M: full-development LinearSVR and persistence — RQ3, RQ4 | stage9_original.ipynb:129<br>stage9_original.ipynb:130 | original Stage 9M trained artifact, convergence/persistence manifest and measurements; test_recovered_protocol.py / test_numerics_models_metrics.py |
| 13.1 | Evidence-backed configuration export — RQ1–RQ4 | read-only explanatory/result presentation | authoritative manifest + run-status JSON + relative evidence CSVs + portable handoff ZIP; test_handoff.py |

## Notebook 01

| Section | Experiment / research question | Original source cells | Generated evidence / verification |
|---|---|---|---|
| 2.1 | Runtime, portable files and fresh execution — RQ1–RQ4 | current_final.ipynb:5<br>current_final.ipynb:6 | input/fold/feature/QR verification outputs and figures; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 2.2 | Original measurements and byte verification — RQ1 | current_final.ipynb:8 | input/fold/feature/QR verification outputs and figures; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 2.3 | Established development and held-out partition — RQ1 | current_final.ipynb:10 | input/fold/feature/QR verification outputs and figures; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 3.1 | Canonical physical and temporal features — RQ1 | current_final.ipynb:13 | input/fold/feature/QR verification outputs and figures; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 3.2 | Numerical checks and grouped-fold compatibility — RQ1, RQ4 | current_final.ipynb:15 | input/fold/feature/QR verification outputs and figures; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 4.1 | Strict import and six-model agreement — RQ1–RQ4 | current_final.ipynb:18 | configuration_agreement.csv and imported_development_manifest.json; test_handoff.py |
| 5.1 | Profile-wise augmented QR and independent identity — RQ4 | current_final.ipynb:21 | input/fold/feature/QR verification outputs and figures; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 5.2 | Training matrix and estimator factories — RQ1, RQ2, RQ3 | current_final.ipynb:23 | input/fold/feature/QR verification outputs and figures; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 6.1 | OLS — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | training_manifest.json / training_times.csv; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 6.2 | k-NN — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | training_manifest.json / training_times.csv; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 6.3 | Extra Trees — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | training_manifest.json / training_times.csv; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 6.4 | MLP — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | training_manifest.json / training_times.csv; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 6.5 | Random Forest — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | training_manifest.json / training_times.csv; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 6.6 | LinearSVR — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | training_manifest.json / training_times.csv; test_features.py / test_sampling_validation.py / test_numerics_models_metrics.py |
| 8.1 | Fixed held-out prediction and checkpoint identity — RQ1, RQ2, RQ3 | current_final.ipynb:29 | 84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 8.2 | Pooled and per-profile metric definitions — RQ2, RQ3, RQ4 | current_final.ipynb:32<br>current_final.ipynb:34 | 84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 9.1 | OLS final evidence — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | 84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 9.2 | k-NN final evidence — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | 84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 9.3 | Extra Trees final evidence — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | 84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 9.4 | MLP final evidence — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | 84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 9.5 | Random Forest final evidence — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | 84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 9.6 | LinearSVR final evidence — RQ2, RQ3, RQ4 | read-only explanatory/result presentation | 84 NPZ checkpoints / overall_metrics.csv / profile_metrics.csv / inference_times.csv; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 10.1 | Session-level RMSE and signed bias — RQ2, RQ3, RQ4 | current_final.ipynb:36 | per-profile diagnostic CSVs and labelled figures; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 10.2 | Bias–variance error decomposition — RQ4 | current_final.ipynb:38 | per-profile diagnostic CSVs and labelled figures; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 10.3 | Measured and predicted temperature histories — RQ2, RQ3, RQ4 | current_final.ipynb:40 | per-profile diagnostic CSVs and labelled figures; test_checkpoints_plotting.py / test_numerics_models_metrics.py |
| 11.1 | Consistency audit, historical comparison and portable outputs — RQ1–RQ4 | current_final.ipynb:42 | run_manifest.json / reference_comparison.csv / compact results ZIP; test_checkpoints_plotting.py / test_numerics_models_metrics.py |

## Figure trace and captions

Development figures are exported to the current UUID run’s figures/ directory. figure_captions.json records notebook, run ID, numbered section, original source notebook/cell, titles and axis labels. Captions and interpretation are in that section; raw tables are retained in the corresponding experiment directories. The hook runs outside the original numerical fitting/prediction timers.

| Final figure | Underlying table or calculation | Meaning / units |
|---|---|---|
| 02_profile_split.png | profile_partition.csv | Session durations (hours), profile IDs and fixed partition |
| 03_ewma_memory.png | original development current and adjusted filters | Current (A), elapsed seconds; diagnostic spans only |
| 04_grouped_cv.png | development_cv_folds.csv | Whole-profile validation-fold membership |
| 08_training_times.png | training_times.csv | Actual seconds under the documented model-dependent fit boundaries |
| 10_accuracy_and_cost.png | overall_metrics.csv + inference_times.csv | RMSE (°C), full inference-loop seconds |
| 11_profile_rmse_bias_heatmaps.png | profile_metrics.csv | Session/model RMSE and signed bias (°C) |
| 11_bias_decomposition.png | profile_bias_diagnostics.csv | Percentage of session MSE from squared mean bias |
| 11_temperature_history.png | original longest held-out profile and saved NPZ predictions | Measured/estimated temperature (°C), elapsed minutes; display-only decimation |

Final source-row indices and model/manifest hashes trace every metric back to the original measurements. No figure implies execution merely because its generating code is present. Some original developmental figures are diagnostic rather than publication figures; retain their explanatory caption and unit interpretation when using them in the report.
