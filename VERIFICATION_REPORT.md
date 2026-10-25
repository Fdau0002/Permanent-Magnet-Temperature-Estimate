# Verification and execution report
## Subsequent student execution in Google Colab

Both production notebooks were completed by the student on 25 October 2026 using Python 3.13.16. Notebook 02 ran first and exported the development selections. Notebook 01 imported the matching manifest, displayed six successful configuration-agreement checks and completed final training and evaluation.

| Check | Recorded outcome | Evidence |
|---|---|---|
| Development execution | Completed; fresh selections exported | reports/executed/ and reports/development_handoff.zip |
| Configuration handoff | Six configurations agreed; matching manifest imported | Saved outputs in both executed notebooks |
| Final evaluation | Six models; 55 development profiles; 14 held-out profiles; 84 prediction checkpoints | Saved Notebook 01 final audit |
| Automated tests | 124 passed; 0 failed; 0 errors; 0 skipped; 1 warning | reports/pytest.html and reports/pytest.xml |
| Source documentation | pdoc HTML regenerated | html/index.html |
| Documentation verification | Passed with no reported structural/link errors | reports/html_verification.json |

The test warning concerns the small synthetic MLP fixture and does not describe convergence of the final production model. The full-run statements above are supported by the saved notebook outputs; this documentation review did not independently rerun the expensive experiments or recompute all saved prediction files.

## Earlier authoring verification record

The remaining sections below preserve the earlier authoring record. References there to pending full execution, empty production outputs or an absent production manifest describe the package before the student's subsequent Colab runs. Earlier environment details, coverage measurements and warning counts belong to that earlier verification.

Prepared 25 October 2026 (Australia/Sydney). Machine-readable run timestamps are retained in UTC in the actual reports. This is a source, targeted-numerical, packaging and reproducibility audit, **not a certification of a completed fresh scientific Run All**.

## Actual successful checks

| Check | Executed outcome | Evidence |
|---|---|---|
| Automated suite | **124 passed; 0 failed; 0 errors; 0 skipped** | reports/pytest.html, pytest.xml and pytest_console.txt |
| Production statement coverage | 95.74% (96% rounded) | reports/coverage.xml; coverage is not scientific experiment coverage |
| Core environment | CPython 3.12.14; NumPy 2.2.6; pandas 2.2.3; SciPy 1.15.3; scikit-learn 1.6.1; scikit-optimize 0.10.2 | reports/source_verification.json |
| Dependency check | All direct pins match; pip check: no broken requirements | source_verification.json |
| Historical-source integrity | Six source notebook hashes unchanged | source_inventory.json and source-verification check |
| Notebook syntax/order | Both notebooks validate; all code cells compile; 100 parent-code blocks retain their numerical statements/order | chapterisation_code_audit.json; notebook-structure tests |
| Original method recovery | Critical GP calls/bounds/objectives/budgets compare to available archive; objective binding matches source AST | recovered_protocol_matches.json; test_recovered_protocol.py |
| Feature/numerical verification | Independent weighted-moment, prefix/reset, sampling, metric, QR and actual Stage 9 Nyström-factory checks pass | Actual pytest report |
| Synthetic GP dependency smoke | Four analytic scalar-quadratic calls, not a production motor-model search | source_verification.json |
| Independent synthetic QR | Maximum prediction difference 9.77e-15, below 1e-10 | source_verification.json |
| Original-data preprocessing | 23 cells executed through original source cell 36 in 7.46 s; stops before model optimisation | reports/preprocessing_check.json |
| Dataset identity/schema | Exact 1,330,816-row/69-profile source; original SHA-256, finite schema and grouped allocation verified | Data integration tests and actual preprocessing outputs |
| Handoff | Actual exporter/loader tested; six constructors, fresh-score selection, missing/altered/incomplete/smoke/active-newer rejection and cross-directory archive transfer pass | test_handoff.py |
| Prediction evidence | Model/profile/source-row/mode/artifact/selection identity and finite shape validated | test_checkpoints_plotting.py |
| Lightweight profiling | Actual production feature/QR functions profiled on 256 rows of development profile 17 | production_profile.txt/json/pstats; separate from historical timings |
| API HTML | 15 actual pdoc HTML pages; current public functions present | docs/html/ and reports/html_verification.json |
| Notebook HTML previews | Two offline MathML previews; 151 development and 84 final equations | docs/notebook_previews/ |
| Local documentation links | 13,872 local file/fragment targets checked; zero errors | html_verification.json |
| Packaging | Version 2.0.0 wheel built; all 14 module bytes match source; separate wheel import succeeds | wheel_validation.json and wheel_import.json |

The original CSV hash is `78f3d150f0f2ad9c5dc7ff24dd12c00d386ad530f48c1589ad24fcd88867d3ad`. Automated public transfer is not newly certified: verification used the supplied original bytes. Manual local-file acquisition and exact hash validation remain the documented alternative.

## What the tests establish

The suite exercises real imported production functions and actual notebook factories. It checks causal prefix invariance/profile resets; independent adjusted moments; source feature parity; exact profile assignments; nested sample identity; extracted grouped objectives versus source; full QR versus independent/direct solves; metric equations and edge cases; estimator training-only scaling/serialization; manifest export/import/selection; current-run completion status; evidence integrity; and rejection of mixed prediction files.

Handoff tests use **explicitly fabricated tables** to exercise complete IO shapes/counts. Their scores are not search results and they are never delivered as production selections. The tiny GP test uses an analytic quadratic. The preprocessing execution uses real raw data and the unchanged original preprocessing code, but stops before Bayesian/model experiments. None of those checks supplies missing full scientific results.

The 15 recorded pytest warnings comprise 14 Matplotlib/pyparsing deprecation warnings and one maximum-iteration warning from the small synthetic MLP serialization fixture. That fixture tests interface/scaling/persistence, not convergence of a production model. Warnings were retained rather than hidden. Full final-model training still stops at the original convergence gate if it warns.

## Educational structure and code preservation

Notebook 02 has 559 cells, including 246 code cells (median 28 lines). Notebook 01 has 210 cells, including 35 code cells (median 12 lines). Chapters/sections/subsections are consistently numbered and substantial experiments use purpose, mathematics, implementation, results and interpretation. Every final model has its own mathematical/training explanation and generated final-result section.

The longest preserved development block is 229 lines: an original nested timed/grouped-fold loop. Such loops and memory-monitor lifecycles remain together to preserve scope and measurement boundaries. No giant stage function was introduced merely to shorten the notebooks. Function docstrings were expanded without changing numerical bodies. AST comparison ignores only function documentation and rejoins partitions of the same invariant run-flag guard; statement order is also checked globally across parent source cells.

## Deliberately unverified until the student’s full Run All

- Complete corrected Bayesian campaigns, fresh selected spans/parameters and all expensive numerical/SVR continuations.
- Full-development convergence, memory feasibility, fitted coefficients and reproducible outcomes of all six final estimators.
- All 84 fresh held-out checkpoints, complete final metrics/figures/costs and model rankings.
- Numerical comparisons of the complete fresh connected workflow with archived historical outputs.
- Live Colab installation/runtime behaviour and long-running resource limits.

The production notebooks contain no fabricated executed outputs, and this package contains no authoritative production manifest. They retain the full original budgets. Notebook 01 requires the newly completed selection and its matching evidence; it does not fall back to historical settings. The preprocessing smoke and tests can be regenerated without changing those production budgets.

## Prior execution and provenance limitation

A development attempt started before execution responsibilities were clarified had progressed into the supplied development searches. It was interrupted before a valid handoff/final evaluation and retained outside this deliverable. Its stopped checkpoint is not a defect-induced scientific result, is not used to claim completion, and supplies no manifest here. No complete corrected production search or final-model training/evaluation was initiated as part of the subsequent lightweight verification.

Historical source variants establish recoverable procedures, not independent authentication of every earliest research session or a complete original random-library/hardware environment. Historical held-out results have already been examined; the partition is retrospectively fixed rather than pristine at project level. These limitations remain explicit in both notebooks and the methodology report.

## Documentation and assessment limits

pdoc HTML was actually generated and structurally/source-content/link checked. A pixel-level browser review was not completed: this host lacked Chromium and its browser download failed. The offline previews contain valid MathML, but no claim of browser-interaction testing or screenshot review is made. Open docs/index.html or docs/html/index.html after extraction to review them.

The separate 5–10-page report, personal reflection, presentation/attendance, GitHub publication/accessibility, meaningful commit history and Moodle submissions remain student responsibilities. No Git operations were performed. The package is ready for review and integration; it cannot certify completion or an assessment grade for those external deliverables.
