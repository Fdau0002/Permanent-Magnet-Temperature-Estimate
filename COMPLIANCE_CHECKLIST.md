# Project-brief compliance and remaining verification

This checklist uses the complete supplied 15-page MMA3001 project brief, including assessment-deliverable pages 8–9 and rubric pages 10–15. The latest attached brief was inspected; its source hash is recorded in `provenance/document_sources.json`. **Prepared**, **tested**, **pending full execution** and **student responsibility** are different statuses. Code execution alone is not a compliance certificate or an assessment mark.

| Requirement | Brief page(s) | Delivered evidence | Status / remaining work |
|---|---|---|---|
| Clear engineering context, importance, intended use and limits | 2, 4, 10 | Both Chapter 1 sections; README; data contract | Prepared; student must be able to explain it |
| Numerical/ML method relevant to mechanical engineering | 2, 4, 7 | Preserved regression, QR, kernel and sensitivity methods | Both student Colab runs completed; saved executed notebooks retained |
| Clearly defined top-level inputs, units, domains and outputs | 4, 10 | DATA_DICTIONARY.md; both feature/data chapters; function docs | Prepared; original-byte/schema integration tests executed |
| Missing/invalid/unsupported input behaviour | 4, 10, 13 | Strict loader/features; error docstrings; invalid-input tests | Tested; no undocumented imputation/clipping |
| Function-level parameters, returns, exceptions and usage | 4, 12 | Actual src/ NumPy-style docstrings; pdoc HTML | Generated and structurally inspected; unit-tested public functions |
| Explain method, assumptions, settings and implementation | 4, 14 | Five-part numbered experimental sections and model mathematics | Explanations prepared; completed numerical outputs available for interpretation |
| Credible alternative/baseline comparison | 5, 14 | Original OLS/k-NN/RF/ET/MLP/SVR/Nyström/LinearSVR sequence | Comparative outputs generated in the completed student runs; engineering interpretation remains for the report |
| Justified validation and metrics | 5, 13 | Profile grouping; mean fold MSE; final pooled/profile metrics | Automated tests passed; final evaluation outputs generated on the 14 held-out profiles |
| Ground-truth/withheld measurements, selection boundary and limits | 5, 13 | Fixed development/test assignment, strict manifest, checkpoint identity | Tested for software integrity; prior held-out inspection disclosed |
| Numerical verification | 2, 5, 7, 13 | Feature identities, QR/direct equivalence, rank and kernel checks | Automated numerical checks passed; development notebook numerical investigations completed |
| Optimisation, sensitivity, stability and scalability | 6, 14 | Original GP searches and Stage 5/9 experiments | Full student development run completed with the preserved experimental budgets |
| Profilers and timing routines | 6 | Preserved timers/RSS; separate cProfile report; PERFORMANCE.md | Earlier lightweight profiling retained; student-run experiment timings available in saved outputs |
| Relevant work/memory/complexity estimates | 6 | PERFORMANCE.md scaling and exact feature-array storage | Prepared; no fabricated hardware FLOP/s or peak-memory measurement |
| Compare baseline and selected solution and justify final design | 6, 14 | Development selections and final comparative tables/figures | Comparative results generated; final engineering justification remains for the written report |
| Clearly organised reproducible repository/README | 2, 7, 8, 11 | src/, notebooks/, tests/, scripts/, guides and manifest contract | Prepared and checked as an integration package |
| Version control during development; meaningful commits/branches | 2, 7, 8, 11 | COMMIT_GUIDE.md only | Student responsibility; existing repository not inspected; no Git operations |
| Accessible final repository link and functional main branch | 8, 11 | Portable package and verification reports | Student must integrate, execute, review and publish an accessible repository |
| Zipped clone submitted via Moodle | 8 | Delivered integration ZIP | Not a clone of the unseen existing repository; student must make/submit the final clone |
| Source code with inline comments and consistent docstrings | 8, 12 | Actual imported modules and readable source cells | Prepared and tested; preserved atomic loops intentionally remain together |
| HTML source documentation | 8, 12 | Generated docs/html/index.html and module pages | Regenerated in Colab; structural/link verification passed |
| Test functions and actual test report | 8, 13 | tests/; real pytest HTML and JUnit XML; coverage XML | Latest Colab run: 124 passed, 0 failed, 0 skipped and 1 warning; HTML/XML reports retained |
| Licence agreement | 8 | Proposed MIT LICENSE and THIRD_PARTY_NOTICES.md | Prepared; reconcile with existing repository ownership/licence before publishing |
| AI tool/task/contribution/verification disclosure | 2, 6, 15 | AI_USE.md; methodology and verification reports | Factual reconstruction disclosure prepared; whole-project contribution requires student review |
| Personal AI reflection, decisions, errors and learning | 6, 15 | AI_USE.md guided reflection prompts | Student responsibility; no invented personal reflection |
| Written individual report, 5–10-page body, references/appendices excluded | 8 | Report structure guidance in REPORT_GUIDE.md; notebook documentation | Separate final written report not certified complete |
| Five-minute presentation plus three-minute questions; appointment attendance | 9 | PRESENTATION_GUIDE.md | Student responsibility; no completed presentation or attendance claim |
| Self-selected-topic approval | 1–2 | No approval record supplied in this package | Student must confirm prior teaching-team approval; not inferred from notebooks |
| Demonstrated individual learning and engineering judgement | 1, 13–15 | Explanations and reflection guidance support preparation | Cannot be certified by an AI-generated package or test count |

## Additional requested deliverables

| User requirement | Evidence / state |
|---|---|
| Notebook 02 executes before Notebook 01 | Role statements, README, Colab guide; strict import gate |
| No historical parameter fallback | Removed production defaults; explicit mapping; source tests |
| Fresh values for every optimised family | Fresh pilot/full/larger search flow; stage reference wiring; exporter reselects minima |
| Original 320/640 budgets and search algorithms | protocol JSON and source AST comparison, no grid replacement |
| Full handoff provenance | Dataset/code/software/folds/parameters/metrics/budgets/seeds/preprocessing/C adjustment/evidence hashes |
| Stop inappropriate runs | Missing/incomplete/active-newer/smoke/altered/wrong-data/wrong-source/wrong-version tests |
| Detailed consistent chapterisation | Consecutive chapter numbering and five-part numbered subsections; tests check hierarchy |
| Every final model explained and individually evaluated | Chapter 6 model-specific mathematics and Chapter 9 generated result sections |
| Figures with labels/captions/source trace | Original labelled plots; source/chapter/run figure index; EXPERIMENT_REGISTER.md |
| Preserve chronology and numerical work | All original parent-code statements/order verified; timing/nested loops retained |
| Do not overwrite historical originals | Byte-identical copies and SHA-256 inventory; source hashes verified |
| Do not perform expensive full runs for the student | Full production runs were performed by the student in Colab; earlier authoring verification used lightweight checks. |

## Completed execution and remaining review

The student completed both full Colab notebook runs. Saved outputs record fresh development selections, the completed handoff, six final trained models and 84 held-out model/profile checkpoints. The subsequent automated suite passed all 124 tests, and regenerated HTML documentation passed structural/link verification.

Completion of execution does not establish every engineering claim. Interpretation of the results, the separate written report, personal AI reflection, presentation, repository publication and submission remain the student's responsibility. Historical held-out inspection remains a disclosed limitation.
