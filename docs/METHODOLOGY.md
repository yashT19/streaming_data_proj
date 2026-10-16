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
- **Input**: Full 30 features (classical preprocessing)
- **Output**: Binary class label (0 or 1) and confidence scores

### Quantum: PennyLane Circuit

- **Qubits**: 4 (matching 4-feature input)
- **Encoding**: Angle encoding of scaled features
- **Layers**: 2-3 parameterized layers with entanglement
- **Optimizer**: Adam (classical optimization of parameters)
- **Input**: 4 reduced features (quantum preprocessing)
- **Output**: Binary class label (0 or 1) and confidence scores

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
