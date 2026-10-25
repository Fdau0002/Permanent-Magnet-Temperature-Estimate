# Source-grounded methodology preservation report

## Scope and evidence hierarchy

The fixed four research questions are reproduced verbatim in both numbered notebooks. The objective remains magnet-temperature regression using profile-disjoint development validation and the original numerical investigations. The corrected package changes organisation, reusable-code boundaries, provenance validation and the computational handoff; it does not replace the original experimental algorithms with a cheaper protocol.

The six unmodified source notebooks are inventoried with SHA-256, code-cell counts and saved-error locations in `provenance/source_inventory.json`. `current_development.ipynb` and `current_final.ipynb` are the supplied revised versions. `historical_scalability.ipynb`, `recovered_full_data_archive.ipynb`, `fresh_reconstruction.ipynb` and `stage9_original.ipynb` provide the recoverable original/restoration lineage. Their own prior restoration statements are evidence, not independent proof of earlier executions or chronological authorship. Full trial histories from every earliest research session cannot be inferred from code or rounded outputs.

The archive supplies actual GP calls, bounds, objective functions and budgets. The current revised development source alone was insufficient: it omitted the larger campaigns. Recovery therefore uses the available historical/full-data sources, with exact source matches recorded in `provenance/recovered_protocol_matches.json`. Cell numbers in this report are **zero-based source-cell indices**, independent of Jupyter execution counters. The chapterised cells retain `source_notebook`, `source_cell`, parent-cell and partition metadata.

## Exact protocol comparison

| Experiment / requirement | Recoverable source evidence | Corrected implementation | Preservation / difference |
|---|---|---|---|
| Source data | Original attached CSV, 1,330,816 rows, 69 profiles, 2 Hz | Exact hash/schema/block checks | Scientific data unchanged; validation added |
| Fixed partition | Fresh source cells 10–18; current final cell 10 | Same stratified profile allocation and locked IDs | 55/14 profiles and 1,055,198/275,618 rows retained |
| Inputs | Fresh 23/25; current final 13 | `features.make_paper_inputs` and causal EWM functions | Equations/order/adjustment/bias/reset preserved; extracted functions imported |
| Grouped CV | Fresh 33/36; final 15 | Same profile qcut strata and shuffled three-fold assignment, seed 42 | No row-random replacement; mean fold MSE retained |
| Preliminary diagnostics | Fresh 38–46 | Chapters 5.2–5.5 | Model-interface, OLS errors, recording starts and outlier-influence investigations retained |
| OLS pilot | Fresh 50/53; archive 47/50 | `DevelopmentStudy.evaluate_ols_cv`, original `gp_minimize` | 8 calls, 4 initial points, EI, seed 42, 80 rows/profile |
| Expanded OLS | Fresh 56/57; archive 53/54 | Original GP ask/tell with fresh pilot observations | 20 additional trials, 8 warm observations, 10 initial points; default acquisition optimiser retained |
| Five-model pilots | Fresh 65–70; archive 60–65 | Original GP ask/tell, reusable original candidate factory | 8 each, 5 initial points, EI/sampling, seed 42, 80 rows/profile |
| Initial exact-SVR scaling | Fresh 73–76; archive 68–71 | Chapter 8.1 | Original changing/fixed-validation designs and timing definitions retained; fresh pilot reference |
| Nested exact-SVR scaling | Fresh 78/80; archive 73/75 | Chapter 8.2; `nested_sample_order` | 80/160/320/640 fitting, 640 validation, 3 folds × 3 repeats = 36 records |
| Full-data QR OLS | Fresh 86/89/91; archive 80/83/85 | `compress_profile_ols` and `DevelopmentStudy.evaluate_full_ols_cv` | Full profile histories, augmented QR, moments and gelsd retained; fresh expanded winner seeds feasibility |
| Full-data OLS BO | Fresh 93/95; archive 87/89 | Chapter 9.3 | 20 total evaluations including fresh feasibility seed; GP/EI/sampling, 6 initial points, seed 42 |
| Larger five-model BO | Fresh 100/102; archive 93/95 | Original common loop and `evaluate_larger_model_cv` | 12 each including freshly rescored pilot seed; GP/EI/sampling, 5 initial points, seed 42 |
| Larger sampling | Fresh 100; archive 93 | Same nested history-before-sampling implementation | SVR 320 fitting/profile; k-NN/RF/ET/MLP 640; validation 640 |
| Training feasibility | Fresh 108/111; archive 100/103 | Chapter 11.1 | Original budgets, timings/RSS and projections retained |
| Stage 9 reference | Fresh 123 replaces old constants in Stage 9 | Larger-search SVR minimum from **this execution** | Intentional requested parameter-handoff correction; no hardcoded old spans/C/epsilon |
| Stage 9B–K | Fresh 125–163; original Stage 9 source | Chapter 12.2–12.11 | Original kernels, scalers, solvers, tolerances, sample sizes, inner/outer comparisons and sensitivity investigations retained |
| Stage 9L–M | `stage9_original.ipynb` 127/129/130 | Chapter 12.12–12.13 | Original full features, converged-minimum C, C×N rule, solver, monitoring and serialization retained |
| Final configurations | Supplied final source loaded `frozen_configurations()` | `handoff.export_manifest` → `load_manifest` → explicit `SELECTED` | Defect corrected: fresh evidence is authoritative; no fallback |
| Final estimator training | Current final 21/23/25 | Same QR/scaled factories, 55 full profiles, original order | Settings preserved; rank-109 fixed expectation relaxed to a diagnostic interval for fresh spans |
| Final inference | Current final 29 | Same 14-profile loop; k-NN batches 64, others 4096 | Predictions unchanged; artifact/selection hashes added to checkpoints |
| Final metrics | Current final 32 and `score` | Same equations/pooled/profile aggregation | Additional checkpoint identity validation only; no score/weighting changes |
| Final interpretation | Current final 34–42 | Same plots, bias decomposition and retrospective reference comparison | Wording discloses prior held-out inspection; old scores never select a model |

## Original search dimensions

| Stage/model | Dimensions and priors |
|---|---|
| Initial OLS spans | Integers [5,30], [31,150], [151,700], [701,3000] |
| Expanded/full/model spans | Integer s1 [5,100], g2 [1,400], g3 [1,1500], g4 [1,6000]; cumulative spans |
| RBF-SVR | C log-uniform [0.001,10]; epsilon log-uniform [0.01,0.3] |
| k-NN | k integer [1,512]; uniform/distance weighting |
| RF / Extra Trees | trees [20,100], depth [5,30], split [2,20], leaf [1,10], bootstrap yes/no |
| MLP | layers [1,3], equal width [4,32], alpha log-uniform [1e-7,1e-2], initial learning rate log-uniform [1e-4,1e-2] |

All Bayesian objectives minimise the **unweighted arithmetic mean of three grouped fold MSE values in original °C²**. Fold scalers fit only fitting profiles. Pilot MLP has max_iter=200, no early stopping. Larger/final MLP has max_iter=400, 10% internal row-based early stopping, patience 15. These differences are retained rather than harmonised. `search.protocol()` records the exact machine-readable contract.

Stage 9I’s five multiplicative C values and Stage 9J/K’s further C sensitivity trials remain finite grids because they were historical numerical sensitivity experiments. They follow the Bayesian RBF reference and do not replace it. Final LinearSVR C is `reference_C * mean_actual_fold_training_rows / 1055198`. The reference is the lowest grouped mean MSE among converged combined Stage 9J/K candidates. It is a heuristic adjustment, not a freshly validated full-data optimum.

## Structural and reliability changes

| Change | Reason and evidence that science is preserved |
|---|---|
| Portable relative paths / dataset retrieval | Removes personal Drive/Colab-state dependency; exact CSV hash prevents changing the input |
| Bind ambient notebook state to `DevelopmentStudy` | Functions use explicit verified development data; AST tests normalise only `self` binding and docstrings to compare original arithmetic |
| Extract shared feature/model/QR/metric functions | Both notebooks import actual production functions; independent tests compare old/new definitions and numerical identities |
| Chapterise Markdown / partition cells | Canonical AST records verify every original parent cell’s statements and order. Consecutive invariant run guards are rejoined for comparison; timed/nested loops stay atomic |
| Read-only final model sections | Display previously computed/imported settings and results; tests reject fit/predict/ask/tell calls in presentation-only cells |
| Fresh reference wiring | Explicitly required correction: earlier current-run selections feed later experiments instead of old constants |
| Active-run completion marker | Prevents an interrupted new attempt from reusing an older complete manifest; no model changes |
| Portable handoff archive | Copies exact manifest/status/evidence bytes; export/import and new-runtime extraction are tested |
| Fail-closed final import | Stops missing/incomplete/wrong-source/wrong-version/wrong-dataset/smoke configurations before training |
| Checkpoint identity checks | Reject mixed model/profile/source-row/manifest predictions before unchanged metric calculations |
| OLS rank diagnostic | Fresh span selections may change rank. Solver remains unchanged; fixed historical rank is not a mathematical invariant |
| Figure trace context | Saves titles, axis labels, run ID, chapter and original source cell outside original numerical timing blocks |

## Historical constants versus fresh outputs

Fresh outputs: all selected spans; all BO-selected model parameters; Stage 5 pilot reference; full-data OLS seed/selection; larger RBF Stage 9 reference; C-grid values derived from that fresh reference; selected LinearSVR reference/full C; validation scores; final fitted coefficients and metrics.

Invariant experimental inputs: source hash/schema/IDs/fold method; seed 42; four spans/108-feature construction; search dimensions/priors/budgets; original prototype/verification inputs; landmark grids; C multipliers; solver losses/tolerances/iteration limits; timing repetitions; final batch sizes. Historical final RMSEs are retained only in an explicitly retrospective comparison after evaluation. They are not acceptance targets.

No production file reads `provenance/historical_configurations.json` as a parameter source. A test fixture’s fabricated 12-row table is software-contract evidence, not a purported real search. No such fixture is included in `results/full/`.

## Remaining limits and unknowable details

Recoverable source code establishes the supplied procedure. It does not reconstruct missing original random-library environments, hardware state, complete earliest trial logs, pre-recording thermal history or unpublished operator decisions. Historical source files include restored variants; literal matches demonstrate agreement with those available artifacts, not independent archival authentication.

The corrected full connected Run All is not certified by this authoring turn. New optima, convergence, full-runtime feasibility, final error/cost/rankings and numerical differences from historical outputs remain to be measured by the student. Exact historical numerical identity is not a goal or a pass criterion.
