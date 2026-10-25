# Change log

## Connected, chapterised reconstruction — 25 October 2026

- Restored recoverable original GP/EI pilot, expanded, full-QR and larger five-model searches; retained source bounds, trial counts, nested sampling and grouped objectives.
- Replaced optimisation-derived historical constants with fresh computational handoffs. Stage 5 takes the current pilot SVR; Stage 9 takes the current larger-search SVR; full-data OLS uses the fresh expanded incumbent.
- Removed the production `frozen_configurations()` fallback. Final training requires the validated development manifest and displays all six agreements.
- Added exact dataset/source/evidence hashes, original budgets/seeds, dependency versions and LinearSVR C×N calculation to the handoff. New active runs block reuse of older selections.
- Added a compact complete handoff archive for independent Colab runtimes and row/model/manifest identity checks when loading final prediction evidence.
- Created consistent chapter/section/subsection numbering, five-part experimental explanations and model-specific mathematics/final-result sections. Partitioned source cells at statement boundaries while retaining original timed/nested loops.
- Added AST/source-preservation tests, handoff rejection/transfer tests, prediction/figure traceability tests and generated pdoc HTML.
- Added execution instructions, source-grounded methodology comparison, experiment/figure register, compliance checklist, AI-use record and integration guidance.
- Cleared production outputs. A prior partial development attempt was interrupted and kept outside the deliverable; it does not certify this corrected workflow. Full Bayesian optimisation and final evaluation are explicitly left to the student.

No Git operations, repository publication or submission steps were performed. Historical source notebooks were not overwritten. Scientific limitations are disclosed rather than silently corrected.
