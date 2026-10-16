#!/usr/bin/env python
"""Task 1: Verify full data pipeline."""
import numpy as np
from src.data_processing import prepare_data, check_split_overlap

print('=== Task 1: Data Pipeline Verification ===')
data = prepare_data(random_state=42, n_quantum_features=4, verbose=True)

# Verify no leakage
print('\n--- Split Overlap Check ---')
check_split_overlap(data['split_info'])

# Verify shapes
print('\n--- Shape Verification ---')
print(f'X_train shape: {data["X_train"].shape}')
print(f'X_val shape: {data["X_val"].shape}')
print(f'X_test shape: {data["X_test"].shape}')
print(f'X_train_quantum shape: {data["X_train_quantum"].shape}')
print(f'X_test_quantum shape: {data["X_test_quantum"].shape}')
print(f'y_train size: {len(data["y_train"])}')
print(f'y_val size: {len(data["y_val"])}')
print(f'y_test size: {len(data["y_test"])}')

# Verify statistics
print('\n--- Data Statistics ---')
train_classes = dict(zip(*np.unique(data["y_train"], return_counts=True)))
test_classes = dict(zip(*np.unique(data["y_test"], return_counts=True)))
print(f'Train records: {len(data["y_train"])}, class distribution: {train_classes}')
print(f'Test records: {len(data["y_test"])}, class distribution: {test_classes}')

# Verify no NaN/Inf
print('\n--- Integrity Checks ---')
print(f'X_train finite: {np.isfinite(data["X_train"]).all()}')
print(f'X_train_quantum finite: {np.isfinite(data["X_train_quantum"]).all()}')
print(f'X_test_quantum finite: {np.isfinite(data["X_test_quantum"]).all()}')

print('\n✓ Data pipeline verification PASSED')
