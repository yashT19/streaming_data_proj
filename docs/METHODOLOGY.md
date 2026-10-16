# Experimental Methodology

*This document will be completed after implementation. It describes the dataset, preprocessing, models, and evaluation approach.*

## Dataset

The Breast Cancer Wisconsin Diagnostic dataset from scikit-learn will be used:
- 569 samples, 30 features
- Binary classification (malignant/benign)
- Stratified train/validation/test split: 60/20/20

## Preprocessing

### Classical Pipeline
- StandardScaler normalization
- Fit on training data only

### Quantum Pipeline
- PCA to reduce to 4 features (capturing ~95% variance)
- StandardScaler normalization
- Fit on training data only

## Models

### Classical: SGDClassifier
- Incremental learning via `partial_fit()`
- Loss function: log_loss
- Random seed for reproducibility

### Quantum: PennyLane Circuit
- 4 qubits, 2 layers
- Angle encoding of 4 features
- Parameterized gates with entanglement
- Adam optimizer

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Training/update/inference timing
- Throughput (records/second)

*Detailed definitions and methodology to follow.*
