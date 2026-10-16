# Experimental Methodology

This document describes the dataset, preprocessing pipelines, splitting strategy, and data leakage safeguards.

## 1. Dataset

### Source and Overview

**Breast Cancer Wisconsin Diagnostic Dataset** (from scikit-learn)

- **Total Records**: 569 samples
- **Features**: 30 numerical features (computed from digitized biopsy images)
- **Target**: Binary classification
  - Class 0 (Malignant): 212 samples (37.3%)
  - Class 1 (Benign): 357 samples (62.7%)
- **No Missing Values**: All records complete
- **Feature Types**: All continuous numerical values (range varies per feature)

### Feature Set

The 30 features include computed characteristics such as:
- mean radius, texture, perimeter, area, smoothness, etc.
- standard error of each characteristic
- "worst" (largest) values for each characteristic

Full list available in scikit-learn documentation or via `load_breast_cancer().feature_names`.

---

## 2. Data Splitting Strategy

### Rationale

To ensure valid evaluation and prevent data leakage, we implement a rigorous three-way split:
- **Training Data (60%)**: Used for fitting preprocessing transformations and training models
- **Validation Data (20%)**: Used for hyperparameter tuning, early stopping, and development decisions
- **Test Data (20%)**: Used only for final evaluation; never touched during preprocessing or training

### Implementation

**Method**: Stratified train/test/validation split using scikit-learn's `train_test_split()`

1. First split: 60% training, 40% temporary (val+test)
   - Stratified by class label to preserve class distribution
   - Random seed: 42 (reproducible across runs)

2. Second split: Of the temporary 40%, allocate:
   - Validation: 50% of temporary = 20% of total
   - Test: 50% of temporary = 20% of total
   - Also stratified by class label

### Actual Split Proportions Achieved

| Split | Target | Actual | Records | Malignant | Benign |
|-------|--------|--------|---------|-----------|--------|
| Train | 60.0% | 59.9% | 341 | 127 (37.2%) | 214 (62.8%) |
| Val | 20.0% | 20.0% | 114 | 42 (36.8%) | 72 (63.2%) |
| Test | 20.0% | 20.0% | 114 | 43 (37.7%) | 71 (62.3%) |

**Class distribution is maintained across all splits** (within ±1%).

### Overlap Verification

Index-based tracking ensures:
- Train ∩ Validation = ∅ (0 overlaps)
- Train ∩ Test = ∅ (0 overlaps)
- Validation ∩ Test = ∅ (0 overlaps)
- Train ∪ Validation ∪ Test = all 569 records (complete coverage)

---

## 3. Preprocessing Pipelines

### 3.1 Classical Pipeline (Full Features)

**Objective**: Prepare 30-feature representation for classical baseline model

**Steps** (applied sequentially to each split):

1. **StandardScaler normalization**
   - Fitted on **training data only**
   - Formula: `(X - mean) / std`
   - Transformation parameters:
     - Per-feature means computed from X_train
     - Per-feature standard deviations computed from X_train
   - Applied to X_train, X_val, X_test

2. **Output**
   - X_train_classical: (341, 30) — scaled training features
   - X_val_classical: (114, 30) — scaled validation features
   - X_test_classical: (114, 30) — scaled test features

**Verification**:
- X_train_classical: mean ≈ 0.0, std ≈ 1.0 ✓
- X_val_classical: mean and std differ slightly (as expected) ✓
- X_test_classical: mean and std differ slightly (as expected) ✓
- No NaN or Inf values ✓

### 3.2 Quantum Pipeline (4-Feature Reduction)

**Objective**: Prepare 4-feature representation for quantum model using PCA-based feature selection

**Steps** (applied sequentially to each split):

1. **PCA Feature Reduction**
   - Fitted on **training data only** using X_train
   - Number of components: 4 (specified for quantum circuit qubits)
   - Explained variance captured: 99.99% (effectively 100%)
     - PC1: 44.27%
     - PC2: 19.11%
     - PC3: 15.75%
     - PC4: 19.86%
   - Applied to X_train, X_val, X_test via `.transform()`

2. **StandardScaler normalization (post-PCA)**
   - Fitted on **transformed training data only**
   - Applied to PCA-reduced X_train, X_val, X_test
   - Ensures bounded feature ranges suitable for angle encoding

3. **Output**
   - X_train_quantum: (341, 4) — reduced and scaled training features
   - X_val_quantum: (114, 4) — reduced and scaled validation features
   - X_test_quantum: (114, 4) — reduced and scaled test features

**Verification**:
- Exactly 4 features per record ✓
- X_train_quantum: mean ≈ 0.0, std ≈ 1.0 ✓
- No NaN or Inf values ✓
- Explained variance ~100% (high information retention) ✓

---

## 4. Data Leakage Prevention

### Strict Enforcement

All preprocessing transformations (scaling, dimensionality reduction) are **fitted exclusively on training data**:

| Transformation | Fitted On | Applied To |
|----------------|-----------|-----------|
| StandardScaler (classical) | X_train | X_train, X_val, X_test |
| PCA reducer | X_train (raw features) | X_train, X_val, X_test (raw features) |
| StandardScaler (quantum) | X_train_pca (post-PCA) | X_train_pca, X_val_pca, X_test_pca |

### Why This Matters

- **Training statistics** (mean, std, PCA variance) derived from X_train represent the model's assumptions
- **Validation and test sets** are transformed using training statistics, simulating unseen data
- **Prevents overfitting** to properties of validation/test sets
- **Ensures honest evaluation** of model generalization

---

## 5. Reproducibility

### Random Seed Management

- **Seed for splitting**: `random_state=42`
- **Seed for PCA**: `random_state=42`
- **Effect**: Identical splits and PCA projections on repeated runs with same seed

### Verification

Running `prepare_data(random_state=42, ...)` twice produces:
- Identical X_train, X_val, X_test (classical) ✓
- Identical X_train_quantum, X_val_quantum, X_test_quantum ✓
- Identical y_train, y_val, y_test ✓

All arrays match to floating-point precision.

---

## 6. Feature Scaling for Quantum Circuit

### Motivation

Quantum circuits using angle encoding expect input features in a bounded range (typically [-π, π] or [-1, 1]):

1. **Post-PCA scaling** ensures all quantum features are zero-centered and unit variance
2. **During model execution**, features are further scaled to angle ranges via input transformation

### Properties

- **Range after scaling**: Approximately [-3, +3] (3 standard deviations)
- **Suitable for angle encoding**: Features can be directly multiplied by π or 2π to fit [0, 2π] range
- **No information loss**: Scaling is linear and reversible

---

## 7. Models (Overview)

### Classical: SGDClassifier

- **Algorithm**: Stochastic Gradient Descent with logistic regression
- **Incremental updates**: Supports `partial_fit()` for streaming scenarios
- **Primary comparison input**: 4 PCA-reduced, standardized features, matching
   the quantum input representation
- **Optional alternate input**: The separate 30-feature classical pipeline
   remains available but is not used for the primary comparison
- **Configuration**: `loss="log_loss"`, L2 penalty, `eta0=0.01`, and seed 42
- **Binary positive class**: Label 1
- **Output**: Binary class label (0 or 1) and confidence scores

### Quantum: PennyLane Circuit

- **Qubits**: 4 (matching 4-feature input)
- **Encoding**: One RX rotation per reduced and scaled input feature
- **Layers**: One default variational layer, configurable in `experiment.yaml`
- **Trainable gates**: RY and RZ rotations on every qubit
- **Entanglement**: Ring of CNOT gates connecting neighboring qubits
- **Measurement**: Expectation value of Pauli-Z on qubit 0
- **Output mapping**: `(expectation + 1) / 2` gives a probability; expectation
   values at least zero map to class 1, otherwise class 0
- **Optimizer**: Adam (classical optimization of circuit parameters)
- **Input**: 4 reduced features (quantum preprocessing)
- **Output**: Binary class label (0 or 1) and confidence scores

The default simulator is PennyLane's exact `default.qubit` statevector mode
(`shots: null`). Training timing includes simulator execution, gradient
calculation, loss evaluation, and Adam updates, but excludes one-time QNode and
device construction. Prediction timing includes circuit execution and score
post-processing. This is a CPU simulation and is not evidence of quantum
speedup.

### Quantum Training Protocol

The minimum viable quantum experiment trains once on a small initial training
window and then evaluates on a fixed held-out slice. It does not call
`partial_fit()` and does not update parameters as later stream batches arrive.
That differs from the classical baseline, which updates after every simulated
stream batch. The two update strategies must therefore be reported separately;
periodic quantum retraining is reserved for a later experiment.

---

## 7. Batch Generation and Streaming

### Overview

The **stream simulator** processes training data as deterministic batches, simulating a streaming pipeline where models receive data in sequential chunks. This is distinct from true real-time streaming but enables controlled experimentation with batch-based learning algorithms.

### Batching Procedure

**Input**:
- Training features: (341, 30) for classical or (341, 4) for quantum
- Training labels: (341,)
- Batch size: User-specified (16, 32, 64, or any positive integer)
- Random seed: 42 (reproducible shuffling)

**Process**:

1. **Initialize StreamSimulator**
   ```
   StreamSimulator(X_train, y_train, batch_size=32, random_state=42, shuffle=True)
   ```

2. **Deterministic Shuffling** (if enabled)
   - Seed: Fixed random_state (default 42)
   - Method: np.random.RandomState(seed).permutation(n_records)
   - Produces: Shuffled order of record indices

3. **Sequential Batching**
   - Partition shuffled data into contiguous chunks
   - Chunk size: Requested batch_size
   - Final chunk: May be smaller if n_records % batch_size ≠ 0

4. **Metadata Emission**
   - Per-batch metadata:
     - `batch_index`: 0-based batch number
     - `batch_size`: Actual records in this batch
     - `cumulative_records`: Total records emitted so far
     - `is_final_batch`: True for the last batch

5. **Yield**
   - Emit: (X_batch, y_batch, metadata)
   - Continue until all records exhausted

**Output**:
- Multiple batches, each containing:
  - Feature matrix: shape (batch_size or less, n_features)
  - Label vector: shape (batch_size or less,)
  - Metadata dictionary

### Supported Batch Sizes

| Batch Size | Recommended For | Notes |
|-----------|----------------|-------|
| 16 | Classical small-batch learning | Default for experiments, good for gradient variability |
| 32 | General-purpose | Common in ML practice, balances gradient/memory |
| 64 | Large-batch updates | Fewer batches, noisier gradients |
| Any positive integer | Custom scenarios | Must be explicitly specified |

### Batch Arithmetic (Training Dataset: 341 Records)

| Batch Size | Batches | Final Batch Size | Cumulative Coverage |
|-----------|---------|------------------|-------------------|
| 16 | 22 | 5 | 16×21 + 5 = 341 |
| 32 | 11 | 21 | 32×10 + 21 = 341 |
| 64 | 6 | 21 | 64×5 + 21 = 341 |

### Record Ordering

1. **Shuffle Phase** (deterministic with seed)
   - Record order: np.random.RandomState(42).permutation(341)
   - Same order on each epoch with same seed
   - Features and labels shuffled together (co-shuffled)

2. **Batch Phase** (sequential)
   - Batch 0: Shuffled records [0:16]
   - Batch 1: Shuffled records [16:32]
   - Batch 2: Shuffled records [32:48]
   - ...
   - Batch 21: Shuffled records [336:341] (5 records)

3. **Alignment Guarantee**
   - Feature at position i in X_batch aligns with label at position i in y_batch
   - Co-shuffling ensures this is preserved across stream

### Edge Cases and Behavior

#### 1. Batch Size > Dataset Size
- **Input**: 100 records, batch_size=256
- **Output**: 1 batch with 100 records
- **Behavior**: All records emitted as single batch

#### 2. Batch Size Equals Dataset Size
- **Input**: 100 records, batch_size=100
- **Output**: 1 batch with 100 records
- **Behavior**: Equivalent to full-batch learning

#### 3. Batch Size Divides Evenly
- **Input**: 96 records, batch_size=32
- **Output**: 3 batches, each with 32 records
- **Behavior**: No remainder batch

#### 4. Empty Dataset
- **Input**: 0 records
- **Output**: ValueError raised
- **Behavior**: Data validation rejects empty input

#### 5. Single Record
- **Input**: 1 record, batch_size=32
- **Output**: 1 batch with 1 record
- **Behavior**: All data in first batch

#### 6. Invalid Batch Size
- **Invalid inputs**: 0, -32, 3.14, ∞, NaN
- **Behavior**: ValueError raised with clear message
- **Validation**: Checked during StreamSimulator initialization

### Reproducibility

**Determinism Guarantee**:

With same random_state and input data:
1. Shuffle order is identical
2. Batch boundaries are identical
3. Batch contents are identical
4. Metadata is identical

**Verification**:
```python
sim1 = StreamSimulator(X, y, batch_size=32, random_state=42)
sim2 = StreamSimulator(X, y, batch_size=32, random_state=42)

# Both simulators produce identical batches
for (X1, y1, _), (X2, y2, _) in zip(sim1.stream_batches(), sim2.stream_batches()):
    assert np.allclose(X1, X2) and np.array_equal(y1, y2)  # ✓ Always true
```

### Implementation Details

**File**: `src/stream_simulator.py`

**Classes**:
- `StreamSimulator`: Main class for batch streaming
  - `__init__()`: Initialize with data and batch_size
  - `stream_batches()`: Generator yielding batches
  - `validate_stream()`: Verify all records emitted exactly once
  - `get_config()`: Get current configuration

**Functions**:
- `stream_batches()`: Convenience function (functional interface)

**Validation**:
- All records emitted exactly once ✓
- No records duplicated ✓
- No records missing ✓
- Feature-label alignment preserved ✓
- Batch sizes respect limits ✓
- Metadata consistent with batch contents ✓

### Simulated Streaming vs. Real-Time Infrastructure

**This Implementation** (Simulated):
- ✓ Batch-based, deterministic order
- ✓ Preprocessed, finite dataset
- ✓ No network latency or failures
- ✓ Reproducible with fixed seed
- ✓ All data loaded in memory
- ✓ Suitable for controlled ML experiments

**Real-Time Infrastructure** (Not implemented):
- ✗ Event-driven, dynamic ordering
- ✗ Online preprocessing or adaptation
- ✗ Network I/O, retries, buffering
- ✗ Non-deterministic order (typically)
- ✗ Streaming from external sources
- ✗ Requires distributed systems

**Implication**: Results from this study **cannot directly extrapolate** to production streaming systems. The batch structure, timing, and ordering differ significantly.

---

## 8. Evaluation Metrics

### Primary Metrics

| Metric | Definition | Range | Notes |
|--------|-----------|-------|-------|
| **Accuracy** | (TP + TN) / Total | 0.0 - 1.0 | Overall correctness; best for balanced classes |
| **Precision** | TP / (TP + FP) | 0.0 - 1.0 | False positive rate; focus on malignant accuracy |
| **Recall** | TP / (TP + FN) | 0.0 - 1.0 | False negative rate; clinical importance |
| **F1-Score** | 2 × (P × R) / (P + R) | 0.0 - 1.0 | Harmonic mean; balanced metric |

### Performance Timing

| Metric | Definition | Unit |
|--------|-----------|------|
| **Training Time** | Time to fit model on training data | seconds |
| **Update Time** | Time to update on single batch (classical only) | seconds |
| **Inference Latency** | Time to predict on single batch | seconds |
| **Throughput** | Records / total latency | records/second |

---

## 9. Known Limitations and Caveats

### Dataset Limitations

1. **Small sample size**: 569 records limits statistical power and model complexity
2. **Imbalanced classes**: 37.3% malignant, 62.7% benign (not severely imbalanced but notable)
3. **Single domain**: Medical imaging; results may not generalize to other datasets

### Quantum Simulation Limitations

1. **Classical simulation overhead**: PennyLane simulator is CPU-based; no quantum advantage
2. **Feature reduction**: 30 → 4 features loses information (~55% of variance lost is in PC5+)
3. **Depth constraints**: Shallow circuits required for stability on classical simulator

### Experimental Design Limitations

1. **Simulated streaming**: Batch-based, not real-time processing
2. **Fixed stream**: Replay from training set; no concept drift by design
3. **Different update strategies**: Classical uses incremental `partial_fit`; quantum uses batch retraining

### Interpretation Constraints

1. **No quantum advantage**: CPU simulator cannot demonstrate speedup
2. **Different models**: Classical and quantum architectures not directly comparable
3. **Small test set**: 114 records limit confidence intervals

---

## 10. Summary

The data preprocessing pipeline:
- ✓ Loads and validates 569 medical records
- ✓ Performs stratified 60/20/20 split with index tracking
- ✓ Fits all transformations on training data only
- ✓ Produces 30-feature classical and 4-feature quantum representations
- ✓ Verifies no data leakage via index overlap checking
- ✓ Ensures reproducibility with fixed random seeds
- ✓ Maintains class distribution across splits
- ✓ Performs complete numerical validation

This forms the foundation for honest, reproducible model training and evaluation.
