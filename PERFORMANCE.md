# Performance, profiling and measurement boundaries

The original numerical experiments retain their timers, repetitions and process-RSS monitoring. These measurements belong to their actual scientific comparisons. `scripts/profile_workflow.py` supplies a **separate** cProfile/tracemalloc diagnostic of the imported feature and QR functions on a 256-row prefix of one development profile. Its fixed diagnostic spans are not optimised settings and do not affect any production manifest.

Run with the original CSV available:

```bash
python scripts/profile_workflow.py
```

Actual outputs are `docs/reports/production_profile.{txt,json,pstats}`. The report lists cumulative costs and function-call counts. The elapsed time includes profiling overhead. Python tracemalloc is not an authoritative peak for all native BLAS/NumPy/scikit-learn allocations. Do not relabel this short diagnostic as a measured full-development memory requirement.

## Why the computational costs differ

| Operation | Useful work/storage scaling | Interpretation limits |
|---|---|---|
| Temporal moments | Roughly O(N×12×4) arithmetic per statistic; 108 output columns | pandas grouping/allocation and repeated candidates can dominate wall time |
| QR compression | Tall-matrix QR is approximately 2nd²−(2/3)d³ FLOPs for n≥d | Actual block/recompression costs, BLAS and rank-aware solve add work |
| Direct OLS | Solve compressed factors with d=109 fitted coefficients | Augmented [1,X,y] has 110 columns; numerical rank may be below 109 |
| Exact RBF kernel | Dense full Gram storage would be O(N²) | LIBSVM uses a kernel cache; this is not a claim that every run allocates a full Gram matrix |
| Nyström map | O(Nm) transformed storage plus landmark-kernel work | More landmarks can improve approximation and increase memory/solver cost |
| Brute-force k-NN | Roughly O(Ntrain×Nquery×108) distance work | Actual sklearn backend, batching, threads and optimised pairwise routines affect cost |
| Forest prediction | Roughly tree count × path depth × query count | Tree layout, leaves, parallelism and feature calculation affect elapsed time |
| MLP | Dense layer matrix multiplies per epoch/query | Early stopping and selected architecture change iteration count and work |

The full 1,055,198×108 float64 feature matrix alone contains 911,691,072 bytes (about 0.849 GiB), excluding targets, raw data, scaler copies, model storage and temporary buffers. This is an exact storage calculation, not a measured peak-RSS claim. For the 109-column OLS design, a tall QR work estimate is approximately 2N×109² FLOPs before lower-order terms and the additional profile/chunk recompressions. Arithmetic-intensity estimates would require knowing the actual memory traffic and cache reuse; no unmeasured hardware FLOP/s or bandwidth number is asserted.

## Comparing existing timers

The original final OLS routine times feature construction and QR compression as part of its reported fit. The other final models time `pipeline.fit` after feature construction. The final inference loop includes feature generation and compressed checkpoint writes as well as predictions. Stage 5/9 fit-only and predict-only timers have their own boundaries. Label each quantity with its definition; a bar chart does not make differently scoped times equivalent.

Timing repeats reuse fixed development folds and parameters; they do not create independent accuracy validation. Process RSS measures the whole process, including prior allocations and allocator state. Validate it against known resident arrays before using it to support memory conclusions. No timing, convergence or memory result for the full corrected workflow is claimed until the student’s actual Run All.
