#!/usr/bin/env python
"""Task 12: Verify streaming integrity."""
import numpy as np
from src.data_processing import prepare_data
from src.stream_simulator import StreamSimulator

print('=== Task 12: Streaming Integrity Check ===')
data = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
X_train, y_train = data['X_train_quantum'], data['y_train']

print(f'\nDataset: {X_train.shape[0]} records, {X_train.shape[1]} features')

# Test different batch sizes
batch_sizes = [16, 32, 64]
for batch_size in batch_sizes:
    print(f'\n--- Batch Size {batch_size} ---')
    simulator = StreamSimulator(X_train, y_train, batch_size=batch_size, random_state=42)
    
    # Validate configuration
    config = simulator.get_config()
    print(f'Configuration: {config}')
    
    # Stream and verify
    total_records = 0
    batch_count = 0
    for X_batch, y_batch, metadata in simulator.stream_batches():
        batch_count += 1
        total_records += X_batch.shape[0]
        
        # Verify alignment
        assert X_batch.shape[0] == y_batch.shape[0], f"Batch {batch_count} misaligned"
        
        # Verify metadata
        assert metadata['batch_size'] == X_batch.shape[0], f"Batch {batch_count} metadata mismatch"
        assert metadata['cumulative_records'] == total_records, f"Batch {batch_count} cumulative mismatch"
    
    print(f'Emitted {total_records} records across {batch_count} batches')
    assert total_records == len(y_train), f"Total records {total_records} != expected {len(y_train)}"
    assert batch_count == simulator.n_batches, f"Batch count {batch_count} != expected {simulator.n_batches}"
    
    # Test determinism
    print(f'Testing determinism...')
    first_run = []
    for X_batch, y_batch, _ in simulator.stream_batches():
        first_run.append((X_batch.copy(), y_batch.copy()))
    
    second_run = []
    simulator2 = StreamSimulator(X_train, y_train, batch_size=batch_size, random_state=42)
    for X_batch, y_batch, _ in simulator2.stream_batches():
        second_run.append((X_batch.copy(), y_batch.copy()))
    
    assert len(first_run) == len(second_run), "Different number of batches"
    for i, (X1, y1) in enumerate(first_run):
        X2, y2 = second_run[i]
        assert np.allclose(X1, X2), f"Batch {i} features differ"
        assert np.array_equal(y1, y2), f"Batch {i} labels differ"
    print(f'✓ Determinism verified')

print('\n✓ Streaming integrity verification PASSED')
