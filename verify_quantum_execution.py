#!/usr/bin/env python
"""Task 14: Verify quantum circuit genuine execution."""
import numpy as np
from src.data_processing import prepare_data
from src.quantum_model import QuantumModel

print('=== Task 14: Quantum Circuit Genuine Execution Check ===')
data = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
X_train, y_train = data['X_train_quantum'], data['y_train']
X_test, y_test = data['X_test_quantum'], data['y_test']

print(f'Dataset: train {X_train.shape[0]}, test {X_test.shape[0]}')

# Create quantum model (with max_input_abs=100.0 as used in experiments)
model = QuantumModel(
    n_qubits=4,
    n_layers=1,
    learning_rate=0.01,
    max_iterations=5,  # Smoke test: minimal iterations
    random_state=42,
    shots=None,  # Exact statevector
    max_input_abs=100.0  # Allow for PCA-transformed features
)
print(f'Quantum model created:')
print(f'  - Qubits: {model.n_qubits}')
print(f'  - Layers: {model.n_layers}')
print(f'  - Params: {model.n_params} (shape: {model.n_layers} x {model.n_qubits} x 2)')
print(f'  - Shots: {model.shots} (exact)')
print(f'  - Optimizer: {model.optimizer_name}')

# Train on initial window
stream_window = min(32, len(y_train))
X_window = X_train[:stream_window]
y_window = y_train[:stream_window]

print(f'\nTraining on initial window: {stream_window} records...')
print(f'Features: {X_window.shape}, Labels: {y_window.shape}')

# Verify features are within expected range
print(f'Feature range: [{X_window.min():.4f}, {X_window.max():.4f}]')
assert X_window.min() >= -100 and X_window.max() <= 100, "Features out of range"

train_result = model.train(X_window, y_window)
print(f'Training complete:')
print(f'  - Time: {train_result["training_time"]:.6f}s')
print(f'  - Iterations: {train_result["iterations"]}')
print(f'  - Final loss: {train_result["final_loss"]:.6f}')
print(f'  - Param update norm: {train_result["parameter_update_norm"]:.6f}')

# Verify loss decreased
print(f'\nLoss history: {[f"{x:.6f}" for x in model.loss_history[:3]]}...')
assert model.loss_history[-1] < model.loss_history[0], "Loss did not decrease"
print(f'✓ Loss decreased')

# Verify parameters were updated
print(f'\nParameter verification:')
param_norm = np.linalg.norm(np.asarray(model.params) - model.initial_params)
print(f'  - Norm of parameter update: {param_norm:.6f}')
assert param_norm > 0, "Parameters not updated"
print(f'✓ Parameters updated')

# Make predictions on test set
print(f'\nMaking predictions on {len(X_test)} test records...')
y_pred, pred_result = model.predict(X_test)
print(f'Predictions:')
print(f'  - Shape: {y_pred.shape}')
print(f'  - Classes: {np.unique(y_pred)}')
print(f'  - Inference time: {pred_result["inference_time"]:.6f}s')

# Verify predictions are binary
assert set(y_pred).issubset({0, 1}), "Predictions not binary"
print(f'✓ Predictions are binary')

# Verify scores/probabilities
scores = pred_result['scores']
probs = pred_result['probabilities']
print(f'  - Score range: [{scores.min():.4f}, {scores.max():.4f}]')
print(f'  - Probability range: [{probs.min():.4f}, {probs.max():.4f}]')
assert scores.min() >= -1.001 and scores.max() <= 1.001, "Scores out of quantum range"
assert probs.min() >= -0.001 and probs.max() <= 1.001, "Probabilities out of [0, 1]"
print(f'✓ Scores and probabilities valid')

# Evaluate on test set
print(f'\nEvaluating on test set...')
eval_result = model.evaluate(X_test, y_test)
metrics = eval_result['metrics']
print(f'Metrics:')
print(f'  - Accuracy: {metrics["accuracy"]:.4f}')
print(f'  - Precision: {metrics["precision"]:.4f}')
print(f'  - Recall: {metrics["recall"]:.4f}')
print(f'  - F1: {metrics["f1"]:.4f}')

# Verify metrics are valid
for key, value in metrics.items():
    assert np.isfinite(value), f"Metric {key}={value} is not finite"
    assert 0.0 <= value <= 1.0, f"Metric {key}={value} is not in [0, 1]"
print(f'✓ All metrics valid')

print('\n✓ Quantum circuit genuine execution PASSED')
