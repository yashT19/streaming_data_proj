#!/usr/bin/env python
"""Task 17: Verify reproducibility check."""
import numpy as np
import json
from pathlib import Path
from src.data_processing import prepare_data
from src.stream_simulator import StreamSimulator
from src.classical_model import ClassicalModel
from src.quantum_model import QuantumModel

print('=== Task 17: Reproducibility Check ===')

# Test 1: Data preparation reproducibility
print('\n--- Test 1: Data Preparation Reproducibility ---')
data1 = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
data2 = prepare_data(random_state=42, n_quantum_features=4, verbose=False)

assert data1['X_train'].shape == data2['X_train'].shape, "X_train shapes differ"
assert np.allclose(data1['X_train'], data2['X_train']), "X_train values differ"
assert np.array_equal(data1['y_train'], data2['y_train']), "y_train values differ"
print('✓ Data preparation is reproducible')

# Test 2: Stream simulator reproducibility
print('\n--- Test 2: Stream Simulator Reproducibility ---')
X_train, y_train = data1['X_train_quantum'], data1['y_train']

stream1 = []
for X_batch, y_batch, _ in StreamSimulator(X_train, y_train, batch_size=32, random_state=42).stream_batches():
    stream1.append((X_batch.copy(), y_batch.copy()))

stream2 = []
for X_batch, y_batch, _ in StreamSimulator(X_train, y_train, batch_size=32, random_state=42).stream_batches():
    stream2.append((X_batch.copy(), y_batch.copy()))

assert len(stream1) == len(stream2), "Different number of batches"
for i, (X1, y1) in enumerate(stream1):
    X2, y2 = stream2[i]
    assert np.allclose(X1, X2), f"Batch {i} features differ"
    assert np.array_equal(y1, y2), f"Batch {i} labels differ"
print('✓ Stream simulator is reproducible')

# Test 3: Classical model reproducibility
print('\n--- Test 3: Classical Model Reproducibility ---')
X_stream, y_stream = X_train[:100], y_train[:100]
X_test, y_test = data1['X_test_quantum'], data1['y_test']

def train_classical_once(X_stream, y_stream, X_test, y_test, seed=42):
    model = ClassicalModel(random_state=seed)
    for X_batch, y_batch, _ in StreamSimulator(X_stream, y_stream, batch_size=32, random_state=seed).stream_batches():
        model.process_batch(X_batch, y_batch, classes=np.unique(y_stream))
    y_pred, _ = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)
    return y_pred, metrics['metrics']

y_pred1, metrics1 = train_classical_once(X_stream, y_stream, X_test, y_test)
y_pred2, metrics2 = train_classical_once(X_stream, y_stream, X_test, y_test)

assert np.array_equal(y_pred1, y_pred2), "Classical predictions differ"
for key in metrics1:
    assert metrics1[key] == metrics2[key], f"Classical metric {key} differs"
print('✓ Classical model training is reproducible')

# Test 4: Quantum model reproducibility
print('\n--- Test 4: Quantum Model Reproducibility ---')

def train_quantum_once(X_stream, y_stream, X_test, y_test, seed=42):
    model = QuantumModel(
        n_qubits=4, n_layers=1, learning_rate=0.01,
        max_iterations=5, random_state=seed, shots=None, max_input_abs=100.0
    )
    window = min(32, len(y_stream))
    model.train(X_stream[:window], y_stream[:window])
    y_pred, _ = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)
    return y_pred, metrics['metrics'], model.loss_history.copy()

y_pred1, metrics1, loss1 = train_quantum_once(X_stream, y_stream, X_test, y_test)
y_pred2, metrics2, loss2 = train_quantum_once(X_stream, y_stream, X_test, y_test)

assert np.array_equal(y_pred1, y_pred2), "Quantum predictions differ"
for key in metrics1:
    assert metrics1[key] == metrics2[key], f"Quantum metric {key} differs: {metrics1[key]} vs {metrics2[key]}"
assert np.allclose(loss1, loss2), "Quantum loss history differs"
print('✓ Quantum model training is reproducible')

# Test 5: Check result files can be loaded and are consistent
print('\n--- Test 5: Result File Consistency ---')
import pandas as pd

raw_df = pd.read_csv('results/raw_results.csv')
summary_df = pd.read_csv('results/summary.csv')
metadata = json.load(open('results/experiment_metadata.json'))

print(f'Raw results: {len(raw_df)} experiments')
print(f'Summary: {len(summary_df)} groups')
print(f'Metadata seed: {metadata[0]["split_seed"]}')

# Verify raw results contain both models
assert 'classical' in raw_df['model_name'].values, "Classical model missing from results"
assert 'quantum' in raw_df['model_name'].values, "Quantum model missing from results"
print('✓ Result files are consistent')

print('\n✓ Reproducibility check PASSED')
print('\n✓✓✓ All execution verification tasks completed successfully ✓✓✓')
