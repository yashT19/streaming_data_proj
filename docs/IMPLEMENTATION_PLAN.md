# Implementation Plan: Quantum vs Classical Streaming Data Analytics

**Project Title**: Quantum vs Classical Approaches for Streaming Data Analytics: An Experimental Study

**Status**: Foundation Phase Complete (Phase 0)

**Last Updated**: 2026-10-16

---

## 1. Executive Summary

This document outlines a complete, reproducible, and well-tested academic project comparing classical and quantum machine learning approaches for simulated streaming data classification. The project aims to answer five core research questions about performance, cost, scalability, and the limitations of quantum approaches in this domain.

The project prioritizes correctness, reproducibility, and honest reporting over optimized performance or claimed quantum advantage.

---

## 2. Research Questions

The project investigates:

1. **Predictive Performance**: How do accuracy, precision, recall, and F1-score compare?
2. **Computational Cost**: How do training, updating, and inference times compare?
3. **Scalability**: How do batch size and stream length affect performance?
4. **Limitations**: What prevents concluding that quantum approaches are universally better?
5. **Concept Drift** (optional): How do models behave under simulated distribution changes?

---

## 3. Architecture Overview

### High-Level Data Flow

```
Dataset (Breast Cancer Wisconsin)
    ↓
[Train/Validation/Test Split (60/20/20)]
    ↓
[Classical Feature Preprocessing] → Classical Model (SGDClassifier)
[Quantum Feature Preprocessing] → Quantum Model (PennyLane Circuit)
    ↓
[Streaming Simulator: Sequential Batches]
    ↓
[Classical Model: Incremental Updates (partial_fit)]
[Quantum Model: Initial Training, Later Inference]
    ↓
[Evaluation: Metrics, Timing, Throughput]
    ↓
[Results Aggregation & Visualization]
```

### Core Modules

| Module | Responsibility | Key Functions |
|--------|-----------------|----------------|
| `data_processing.py` | Dataset loading, stratified split, preprocessing | `load_dataset()`, `preprocess()`, `validate_split()` |
| `stream_simulator.py` | Deterministic batch streaming | `StreamSimulator`, `yield_batches()` |
| `classical_model.py` | SGDClassifier wrapper with timing | `ClassicalModel.train()`, `.predict()`, `.update()` |
| `quantum_model.py` | PennyLane circuit classifier | `QuantumModel.train()`, `.predict()`, circuit definitions |
| `evaluation.py` | Metrics computation | `compute_metrics()`, `confusion_matrix()`, `timing()` |
| `experiment_runner.py` | Orchestration of experiments | `run_experiment()`, `run_batch_sensitivity()` |
| `plotting.py` | Result visualization | `plot_accuracy()`, `plot_timing()`, `plot_throughput()` |
| `reproducibility.py` | Seeding and determinism | `set_seeds()`, `validate_reproducibility()` |

---

## 4. Implementation Phases

### Phase 0: Environment Inspection ✓ COMPLETE
- [x] Inspect Python version, OS, memory
- [x] Check for existing virtual environment
- [x] Verify quantum framework availability
- [x] Identify existing user work
- [x] Create implementation plan

**Status**: Complete. Python 3.14.0, NumPy, Pandas, scikit-learn available. PennyLane and Qiskit not yet installed.

---

### Phase 1: Project Structure ✓ COMPLETE
- [x] Create directory hierarchy
- [x] Initialize Git repository
- [x] Add `.gitignore`
- [x] Create `requirements.txt` and `pyproject.toml`
- [x] Create `README.md` and configuration files
- [x] Establish documentation templates

**Status**: Complete.

---

### Phase 2: Data Preparation (PENDING)
**Objectives**:
1. Load Breast Cancer Wisconsin dataset from scikit-learn
2. Perform stratified train/validation/test split (60/20/20)
3. Implement classical preprocessing (StandardScaler)
4. Implement quantum preprocessing (PCA to 4 features, then scaling)
5. Validate no data leakage (fit transformations on train data only)
6. Create reproducibility checks

**Deliverables**:
- `src/data_processing.py`: Full implementation
- `tests/test_data_processing.py`: Comprehensive tests
- Validation of correct train/val/test separation

**Timeline**: ~3-4 hours (accounting for testing and edge cases)

**Risks**:
- PCA fitting on wrong data subset → data leakage
- Missing value handling
- Feature scaling applied before split

---

### Phase 3: Stream Simulator (PENDING)
**Objectives**:
1. Design configurable batch streamer
2. Support batch sizes 16, 32, 64
3. Preserve record order after shuffle
4. Handle incomplete final batches
5. Track emitted records and cumulative count
6. Validate no record duplication or loss

**Deliverables**:
- `src/stream_simulator.py`: Streaming logic
- `tests/test_stream_simulator.py`: Edge case tests
- Verification of record count and order

**Timeline**: ~2-3 hours

**Risks**:
- Off-by-one errors in batch boundary detection
- Incorrect handling of final incomplete batch

---

### Phase 4: Classical Baseline (PENDING)
**Objectives**:
1. Implement `SGDClassifier` wrapper with `partial_fit()`
2. Supply all classes on first batch
3. Measure update time separately from inference
4. Record cumulative training time
5. Support reproducible runs with fixed seed

**Deliverables**:
- `src/classical_model.py`: Model wrapper
- `tests/test_classical_model.py`: Unit tests
- Smoke test verifying training and prediction

**Timeline**: ~2-3 hours

**Risks**:
- Timing instrumentation overhead
- Incorrect class initialization on first batch

---

### Phase 5: Quantum Classifier (PENDING)
**Objectives**:
1. Implement 4-qubit variational quantum circuit in PennyLane
2. Angle encoding of 4 normalized input features
3. Parameterized trainable gates with entanglement
4. Measurement-based or expectation-value output
5. Classical optimizer (Adam)
6. Separate training from inference timing
7. Fixed random seed for reproducibility

**Deliverables**:
- `src/quantum_model.py`: Circuit and classifier
- `tests/test_quantum_model.py`: Circuit and training tests
- Documentation of circuit architecture
- Timing verification

**Timeline**: ~4-5 hours (quantum framework learning curve)

**Risks**:
- Framework incompatibilities or version conflicts
- Optimization not converging on small datasets
- Simulation taking unexpectedly long

---

### Phase 6: Experiments (PENDING)
**Experiments to implement**:

#### Experiment A: Performance Comparison
- Train both models on same train data
- Evaluate on same held-out test set
- Compute accuracy, precision, recall, F1, confusion matrix
- Run once (no repetition needed for single training)

#### Experiment B: Batch-Size Sensitivity
- Batch sizes: 16, 32, 64 (where valid)
- Classical: measure per-batch update time and total processing time
- Quantum: measure inference latency for each batch size
- Repetitions: 3 (report mean ± std)

#### Experiment C: Stream-Length Sensitivity
- Stream lengths: 100, 200, 400 records (from training set)
- Evaluate with fixed test set
- Measure impact on accuracy and final cumulative time
- Repetitions: 3

#### Experiment D: Concept Drift (Optional)
- Create controlled synthetic drift (e.g., gradual feature shift)
- Measure performance degradation over time
- Separate from main results with clear labeling

**Deliverables**:
- `src/experiment_runner.py`: Full orchestration
- Raw result CSV/JSON files in `results/`
- Summary table with aggregated metrics

**Timeline**: ~5-8 hours (including experiment coordination and debugging)

**Risks**:
- Quantum optimization timeout
- Timing variance makes statistical significance unclear
- Insufficient records for meaningful stream lengths

---

### Phase 7: Metrics & Benchmarking (PENDING)
**Metrics to track**:

| Metric | Definition | Unit |
|--------|-----------|------|
| Accuracy | TP+TN / Total | % or ratio |
| Precision | TP / (TP+FP) | % or ratio |
| Recall | TP / (TP+FN) | % or ratio |
| F1-Score | 2 × (P × R) / (P + R) | ratio |
| Train Time | Initial model training | seconds |
| Update Time | Per-batch incremental update (classical only) | seconds |
| Inference Latency | Time to predict one batch | seconds |
| Throughput | Records / (batch latency + data prep) | rec/sec |
| Peak Memory | Maximum process memory during run | MB |

**Methodology**:
- Use `time.perf_counter()` for high-resolution monotonic timing
- Record cold-start vs. warm inference separately
- Separate circuit construction from measurement
- Repeat 3 times, report mean ± std dev
- Do not compare warmed-up classical against cold quantum

**Deliverables**:
- Metric definitions in `docs/METHODOLOGY.md`
- Computed metrics saved in results files
- Timing reproducibility validated

**Timeline**: ~2-3 hours (embedded in experiment runs)

---

### Phase 8: Results Storage & Visualization (PENDING)
**Output formats**:

**Raw Results CSV/JSON**: One row per experiment trial
```
experiment_id, model, dataset, batch_size, stream_length, accuracy, precision, recall, f1, train_time_sec, inference_latency_sec, throughput_rec_sec, run_date, framework_version, ...
```

**Summary Table**: Aggregated by model and experiment

**Plots**:
1. Accuracy comparison (bar chart)
2. Timing vs. batch size (line plot)
3. Throughput comparison (bar chart)
4. Optional: Concept drift performance over time

**Deliverables**:
- `results/raw_results.csv` or `.json`
- `results/summary.csv`
- `results/plot_accuracy.png`
- `results/plot_timing.png`
- `results/plot_throughput.png`

**Timeline**: ~2-3 hours

---

### Phase 9: Automated Testing (PENDING)
**Test coverage**:

| Test Suite | Purpose | Count |
|----------|---------|-------|
| `test_data_processing.py` | Dataset loading, splits, leakage | ~12 tests |
| `test_stream_simulator.py` | Batching, edge cases, record counting | ~10 tests |
| `test_classical_model.py` | Training, prediction, reproducibility | ~8 tests |
| `test_quantum_model.py` | Circuit construction, training, predictions | ~12 tests |
| `test_evaluation.py` | Metric calculations, edge cases | ~8 tests |
| `test_experiment_runner.py` | Orchestration, output schemas | ~6 tests |

**Test strategy**:
- Unit tests use small synthetic datasets
- Avoid full quantum optimization in unit tests
- Separate fast smoke tests from slow integration tests
- Reproducibility tests verify identical outputs on same config

**Deliverables**:
- Full pytest suite in `tests/`
- Smoke test command
- CI/CD ready (can be integrated with GitHub Actions later)

**Timeline**: ~4-5 hours

---

### Phase 10: Dependencies & Documentation (PENDING)
**Deliverables**:

- Clear `requirements.txt` with pinned versions
- Virtual environment setup instructions
- Commands for setup, testing, running experiments
- Troubleshooting guide in README

**Timeline**: ~1-2 hours

---

### Phase 11: Final Documentation (PENDING)

**Documents to complete**:

1. **`docs/METHODOLOGY.md`**:
   - Dataset description and split strategy
   - Preprocessing details for classical and quantum
   - Streaming simulation protocol
   - Classical model algorithm and hyperparameters
   - Quantum circuit architecture and training
   - Evaluation metrics definitions
   - Timing methodology

2. **`docs/RESULTS.md`**:
   - Actual measured results from experiments
   - Summary tables and interpretation
   - Plots and figures
   - Notable findings and anomalies
   - Performance comparison summary

3. **`docs/LIMITATIONS.md`**:
   - Simulated vs. real-time streaming
   - Small dataset size and generalization limits
   - Quantum feature reduction (4 features)
   - Classical simulator overhead
   - Different update strategies (incremental vs. batch)
   - Hardware and software dependencies
   - Why this does NOT establish quantum speedup

**Timeline**: ~2-3 hours

---

## 5. Expected Outputs

### File Structure After Completion

```
quantum-streaming-study/
├── README.md                              # Project overview
├── requirements.txt                       # Pinned dependencies
├── pyproject.toml                         # Project metadata
├── .gitignore                             # Git ignore rules
├── configs/
│   └── experiment.yaml                    # Experiment configuration
├── src/
│   ├── __init__.py
│   ├── data_processing.py                 # Dataset and preprocessing
│   ├── stream_simulator.py                # Batch streaming
│   ├── classical_model.py                 # SGDClassifier wrapper
│   ├── quantum_model.py                   # PennyLane circuit
│   ├── evaluation.py                      # Metrics computation
│   ├── experiment_runner.py               # Experiment orchestration
│   ├── plotting.py                        # Visualization
│   └── reproducibility.py                 # Seeding and reproducibility
├── tests/
│   ├── test_data_processing.py
│   ├── test_stream_simulator.py
│   ├── test_classical_model.py
│   ├── test_quantum_model.py
│   ├── test_evaluation.py
│   └── test_experiment_runner.py
├── results/
│   ├── .gitkeep
│   ├── raw_results.csv                    # Experiment trial data
│   ├── summary.csv                        # Aggregated results
│   ├── plot_accuracy.png
│   ├── plot_timing.png
│   ├── plot_throughput.png
│   └── plot_drift.png                     # If drift experiment runs
└── docs/
    ├── IMPLEMENTATION_PLAN.md             # This document
    ├── METHODOLOGY.md                     # Experimental methodology
    ├── RESULTS.md                         # Actual measured results
    └── LIMITATIONS.md                     # Limitations and caveats
```

### Key Results to Generate

- Accuracy and F1 comparison by model
- Processing time vs. batch size
- Throughput comparison (records/second)
- Stream length effect on accuracy
- Optional: concept drift performance curves

---

## 6. Verification Strategy

### Checkpoint Validation

**Checkpoint 1 (Environment & Plan)** ✓
- [x] Environment inspected
- [x] Implementation plan created
- [x] Directory structure established
- [x] Git repository initialized

**Checkpoint 2 (Data & Stream)**
- [ ] Dataset loads without errors
- [ ] Train/val/test splits are correct
- [ ] No data leakage detected
- [ ] Stream simulator produces expected batches
- [ ] All tests pass: `pytest tests/test_data_processing.py tests/test_stream_simulator.py`

**Checkpoint 3 (Classical Model)**
- [ ] SGDClassifier trains on first batch
- [ ] Model updates on subsequent batches
- [ ] Predictions have correct shape and labels
- [ ] Timing measurements are valid
- [ ] All tests pass: `pytest tests/test_classical_model.py`

**Checkpoint 4 (Quantum Model)**
- [ ] Quantum circuit constructs successfully
- [ ] Circuit has correct number of qubits (4)
- [ ] Training completes on small dataset
- [ ] Predictions have correct shape and labels
- [ ] Circuit outputs are finite and valid
- [ ] All tests pass: `pytest tests/test_quantum_model.py`

**Checkpoint 5 (Integration)**
- [ ] Both models evaluate on same test set
- [ ] Output schema validated
- [ ] No data leakage between train/test
- [ ] Results serialize to CSV/JSON correctly

**Checkpoint 6 (Experiments)**
- [ ] Experiment A runs and produces metrics
- [ ] Experiment B shows batch size sensitivity
- [ ] Experiment C shows stream length sensitivity
- [ ] Plots generated from real results
- [ ] Timing anomalies investigated and documented

**Checkpoint 7 (Quality Assurance)**
- [ ] Full test suite passes (or failures documented)
- [ ] Smoke test completes successfully
- [ ] End-to-end experiment runs (or marked BLOCKED)
- [ ] Documentation complete
- [ ] No fabricated results or claims

---

## 7. Computational Budget & Risks

### Estimated Runtime

| Phase | Task | Estimate | Notes |
|-------|------|----------|-------|
| 2 | Data prep + tests | 3-4 hrs | Includes debugging |
| 3 | Stream simulator + tests | 2-3 hrs | Edge cases |
| 4 | Classical model + tests | 2-3 hrs | Timing instrumentation |
| 5 | Quantum model + tests | 4-5 hrs | Framework learning |
| 6 | Experiments (smoke) | 1-2 hrs | Quick validation |
| 6 | Experiments (full) | 10-20 hrs | 3 reps × 3 configs × quantum opt |
| 7 | Testing suite | 4-5 hrs | Comprehensive coverage |
| 8 | Plotting & storage | 2-3 hrs | Results serialization |
| 9 | Docs & final audit | 3-4 hrs | Complete documentation |
| **Total** | | **~35-50 hours** | Assuming no major blockers |

### Resource Requirements

- **Memory**: ~2-4 GB (quantum simulator + dataset)
- **CPU**: 4+ cores recommended
- **Disk**: ~500 MB for code, dependencies, and results
- **Network**: Required only for initial pip install

### Key Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| PennyLane incompatible with Python 3.14 | HIGH | Test immediately; fallback to Qiskit if needed |
| Quantum optimization timeout | HIGH | Reduce max iterations and circuit depth |
| Data leakage in preprocessing | HIGH | Rigorous validation in tests and code review |
| Insufficient test records for meaningful streams | MEDIUM | Use available data; document limitations |
| Timing variance obscures performance differences | MEDIUM | Report mean ± std; run multiple repetitions |
| Memory pressure from quantum sim | LOW | Reduce batch size or circuit depth |

---

## 8. Known Limitations

### Fundamental Constraints

1. **Simulated Streaming**: Not real-time; batch-oriented processing
2. **Small Dataset**: Breast Cancer Wisconsin (569 records) → limited generalization
3. **Reduced Features**: Quantum model uses 4 features (PCA); classical uses all features in separate experiment
4. **CPU-only Simulator**: No quantum speedup claim possible
5. **Different Update Strategies**: Classical is incremental; quantum is batch-trained
6. **Short Training Horizons**: Limited to hundreds of records

### Expected Limitations in Results

- Quantum model may underperform classical baseline on this small, classical dataset
- Quantum advantage requires very specific problem structures not present in tabular data
- Timing measurements depend on machine, Python version, and simulator state
- Results are not generalizable to production systems or quantum hardware

---

## 9. Acceptance Criteria (Go/No-Go)

Project is **COMPLETE** only when **ALL** criteria are met:

### Must Have ✓ Will Verify
- [ ] Python 3.10+ environment confirmed
- [ ] All core dependencies install without error
- [ ] Dataset loads and preprocessing passes tests
- [ ] Train/val/test split verified correct (no overlaps, correct sizes)
- [ ] Stream simulator emits all intended training records
- [ ] Classical model trains and predicts successfully
- [ ] Quantum circuit executes on selected simulator
- [ ] Quantum model completes training and produces valid predictions
- [ ] Both models evaluated on same held-out test set
- [ ] Metrics and timings computed from actual runs (no fabrication)
- [ ] Raw results saved in CSV/JSON format
- [ ] Summary results table generated
- [ ] Required plots generated from saved results
- [ ] Unit tests pass (or failures documented with reason)
- [ ] Smoke test completes successfully (or marked BLOCKED)
- [ ] Documentation complete and accurate

### Must NOT Do
- [ ] Do not claim quantum speedup or advantage
- [ ] Do not compare pre-warmed classical against cold quantum
- [ ] Do not fit preprocessing on test data
- [ ] Do not manually enter or fabricate any metric value
- [ ] Do not suppress relevant warnings without documentation

---

## 10. Next Steps (After Phase 0)

### Immediate Actions
1. Install PennyLane and verify simulator imports
2. Implement `src/data_processing.py` with full test coverage
3. Implement `src/stream_simulator.py` with deterministic batch logic
4. Run Checkpoint 2 validation
5. Proceed to Phase 4 only after Checkpoint 3 passes

---

## 11. Checkpoint Recording

### Checkpoint 1: Environment & Foundation ✓ COMPLETE
**Date**: 2026-10-16
**Status**: PASS
**Summary**:
- Python 3.14.0 installed
- NumPy, Pandas, scikit-learn available
- Git repository initialized and linked to GitHub
- Project structure created
- Dependencies configured in requirements.txt and pyproject.toml
- Initial documentation in place

**Actions Taken**:
1. Initialized Git repository at `c:\Users\tippa\Desktop\streaming data\`
2. Added GitHub remote: `https://github.com/yashT19/streaming_data_proj.git`
3. Created directory structure (configs, src, tests, notebooks, results, docs)
4. Created `.gitignore`, `requirements.txt`, `pyproject.toml`
5. Created `README.md` with comprehensive project overview
6. Created `configs/experiment.yaml` with experiment parameters
7. Created this IMPLEMENTATION_PLAN.md document

**Blockers**: None at this stage

**Next**: Install dependencies and verify PennyLane/Qiskit availability

---

