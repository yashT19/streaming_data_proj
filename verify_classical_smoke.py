#!/usr/bin/env python
"""Task 13: Verify classical model smoke run."""
import numpy as np
from src.data_processing import prepare_data
from src.stream_simulator import StreamSimulator
from src.classical_model import ClassicalModel

print('=== Task 13: Classical Model Smoke Run ===')
data = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
X_train, y_train = data['X_train_quantum'], data['y_train']
X_test, y_test = data['X_test_quantum'], data['y_test']

print(f'Dataset: train {X_train.shape[0]}, test {X_test.shape[0]}')

# Create model
model = ClassicalModel(random_state=42)
print(f'Model created: {model.model}')

# Stream training
batch_size = 32
stream_length = min(100, len(y_train))
X_stream = X_train[:stream_length]
y_stream = y_train[:stream_length]

print(f'\nStreaming {stream_length} records in batches of {batch_size}...')
batch_count = 0
total_train_time = 0.0

for X_batch, y_batch, metadata in StreamSimulator(X_stream, y_stream, batch_size=batch_size, random_state=42).stream_batches():
    if batch_count == 0:
        # First batch: initialize with class list
        result = model.process_batch(X_batch, y_batch, classes=np.unique(y_stream))
        print(f'Batch 0: {X_batch.shape[0]} records, time={result["update_time"]:.6f}s')
    else:
        result = model.update(X_batch, y_batch)
        if batch_count <= 2 or batch_count == metadata['batch_index']:
            print(f'Batch {batch_count}: {X_batch.shape[0]} records, time={result["update_time"]:.6f}s')
    
    batch_count += 1
    total_train_time = result['cumulative_training_time']

print(f'Total batches processed: {batch_count}, total time: {total_train_time:.6f}s')

# Make predictions on test set
print(f'\nMaking predictions on {len(X_test)} test records...')
y_pred, timing = model.predict(X_test)
print(f'Predictions shape: {y_pred.shape}, inference time: {timing["inference_time"]:.6f}s')

# Evaluate
print(f'\nEvaluating on test set...')
eval_result = model.evaluate(X_test, y_test)
metrics = eval_result['metrics']
print(f'Accuracy: {metrics["accuracy"]:.4f}')
print(f'Precision: {metrics["precision"]:.4f}')
print(f'Recall: {metrics["recall"]:.4f}')
print(f'F1: {metrics["f1"]:.4f}')

# Verify metrics are finite and in [0, 1]
for key, value in metrics.items():
    assert np.isfinite(value), f"Metric {key}={value} is not finite"
    assert 0.0 <= value <= 1.0, f"Metric {key}={value} is not in [0, 1]"

print(f'\nModel stats: {model.get_stats()}')
print('\n✓ Classical model smoke run PASSED')
