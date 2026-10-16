# Checkpoint 4: Classical Incremental Baseline - COMPLETE

**Date**: 2026-10-16  
**Status**: PASS  
**Phase**: Phase 4 - Classical Incremental Model

## Implementation

Implemented `src/classical_model.py` with a documented `SGDClassifier` baseline:

- `loss="log_loss"`
- `penalty="l2"`
- `eta0=0.01`
- `random_state=42`
- `partial_fit()` for every sequential stream batch
- Complete class list supplied on the first update
- Four-feature prepared representation used for the primary comparison
- Per-update and cumulative training timing
- Separate fixed-test inference timing
- Accuracy, precision, recall, F1-score, and confusion matrix
- Binary positive label explicitly set to class `1`
- Structured evaluation and metadata without test-set updates

Added:

- `tests/test_classical_model.py` with 4 regression tests
- `verify_classical_model.py` for end-to-end and reproducibility checks
- Classical baseline results in `docs/RESULTS.md`

## Verification Commands

```text
python -m pytest tests/test_classical_model.py -q
python verify_classical_model.py
```

Results:

- Classical model tests: **4 passed**
- Training records processed: **341 / 341**
- Stream batches: **11** at batch size 32
- Held-out test records: **114**
- Prediction length: **114**
- Repeated predictions and metrics: **identical**
- Invalid-input checks: **3 / 3 passed**
- NaN/infinite metric checks: **passed**
- Warnings: **none**

## Observed Baseline Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8772 |
| Precision | 0.8904 |
| Recall | 0.9155 |
| F1-score | 0.9028 |
| Confusion matrix | `[[35, 8], [6, 65]]` |
| Training/update time | 0.01151410 seconds |
| Test inference time | 0.00019680 seconds |

No quantum model was implemented or run in this checkpoint.