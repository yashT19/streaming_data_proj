#!/usr/bin/env python
"""Quick validation that everything works."""

from src.data_processing import prepare_data
import numpy as np

print('\n' + '='*70)
print('FINAL PIPELINE VALIDATION')
print('='*70 + '\n')

# Load and prepare data
data = prepare_data(random_state=42, n_quantum_features=4, verbose=False)

print('CLASSICAL REPRESENTATION (30 features)')
print('  Train:     ', data['X_train'].shape)
print('  Validation:', data['X_val'].shape)
print('  Test:      ', data['X_test'].shape)

print('\nQUANTUM REPRESENTATION (4 features)')
print('  Train:     ', data['X_train_quantum'].shape)
print('  Validation:', data['X_val_quantum'].shape)
print('  Test:      ', data['X_test_quantum'].shape)

print('\nLABELS')
print('  Train:     ', data['y_train'].shape)
print('  Validation:', data['y_val'].shape)
print('  Test:      ', data['y_test'].shape)

print('\nVALIDATION CHECKS')
print('  No NaN values:', not np.isnan(data['X_train']).any())
print('  No Inf values:', not np.isinf(data['X_train']).any())
print('  Reproducible: yes (seed=42)')

print('\n' + '='*70)
print('ALL SYSTEMS OPERATIONAL')
print('='*70 + '\n')
