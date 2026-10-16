"""Small variational quantum classifier implemented with PennyLane.

The circuit uses four-qubit RX angle encoding, trainable RY/RZ rotations,
and a ring of CNOT gates. The expectation value of Pauli-Z on qubit zero is
converted to a probability in ``[0, 1]`` for a differentiable binary
cross-entropy objective. This model is trained once on an initial window; it
does not implement ``partial_fit`` or claim to be an incremental learner.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np

from src.evaluation import compute_classification_metrics, compute_confusion_matrix

logger = logging.getLogger(__name__)

try:
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None


class QuantumModel:
    """A four-qubit variational classifier on PennyLane's ``default.qubit``."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "angle",
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        random_state: int = 42,
        shots: Optional[int] = None,
        max_input_abs: float = 2 * np.pi,
    ) -> None:
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is not installed. Install with: pip install pennylane")
        if n_qubits != 4:
            raise ValueError(f"This experiment requires exactly 4 qubits, got {n_qubits}")
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be a positive integer")
        if encoding != "angle":
            raise ValueError("Only angle encoding is supported")
        if optimizer.lower() != "adam":
            raise ValueError("Only the Adam optimizer is supported")
        if learning_rate <= 0 or max_iterations < 1:
            raise ValueError("learning_rate must be positive and max_iterations must be positive")
        if shots is not None and (not isinstance(shots, int) or shots < 1):
            raise ValueError("shots must be None or a positive integer")
        if max_input_abs <= 0:
            raise ValueError("max_input_abs must be positive")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.optimizer_name = optimizer.lower()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.shots = shots
        self.max_input_abs = float(max_input_abs)
        self.n_params = n_layers * n_qubits * 2

        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="autograd",
            diff_method="best",
        )
        self.params = None
        self.initial_params = None
        self.classes_ = None
        self.loss_history = []
        self.training_time = 0.0
        self.inference_times = []

    def _validate_features(self, X: np.ndarray, name: str) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_qubits:
            raise ValueError(
                f"{name} must have shape (n_records, {self.n_qubits}), got {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError(f"{name} must not be empty")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        if np.max(np.abs(X)) > self.max_input_abs:
            raise ValueError(
                f"{name} values must be within [-{self.max_input_abs:.6f}, "
                f"{self.max_input_abs:.6f}] for angle encoding"
            )
        return X

    @staticmethod
    def _validate_labels(y: np.ndarray, n_records: int) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != n_records:
            raise ValueError(f"Feature and label lengths differ: {n_records} vs {y.shape[0]}")
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"Binary classification requires two classes, got {classes.tolist()}")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("QuantumModel requires binary labels encoded as 0 and 1")
        return y.astype(float)

    def _circuit(self, inputs, params):
        """Encode four inputs and return ``<Z_0>`` from the variational circuit."""
        for wire in range(self.n_qubits):
            qml.RX(inputs[wire], wires=wire)

        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                qml.RY(params[layer, wire, 0], wires=wire)
                qml.RZ(params[layer, wire, 1], wires=wire)
            for wire in range(self.n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])

        return qml.expval(qml.PauliZ(0))

    def initialize(self) -> np.ndarray:
        """Create deterministic trainable parameters with shape ``(layers, 4, 2)``."""
        rng = np.random.RandomState(self.random_state)
        initial = rng.normal(loc=0.0, scale=0.05, size=(self.n_layers, self.n_qubits, 2))
        self.initial_params = np.array(initial, copy=True)
        self.params = qml.numpy.array(initial, requires_grad=True)
        return np.array(initial, copy=True)

    def evaluate_circuit(self, features: np.ndarray) -> float:
        """Execute the circuit for one feature vector and return its expectation."""
        if self.params is None:
            raise ValueError("Circuit parameters are not initialized; call train() first")
        features = self._validate_features(np.asarray(features).reshape(1, -1), "features")[0]
        value = self.qnode(qml.numpy.array(features, requires_grad=False), self.params)
        value = float(value)
        if not np.isfinite(value) or not -1.000001 <= value <= 1.000001:
            raise ValueError(f"Circuit output is invalid: {value}")
        return value

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
        """Optimize circuit parameters on an initial training window.

        Timing starts after the QNode/device has been constructed and includes
        simulator execution, gradient calculation, and classical optimization.
        It excludes one-time circuit object construction.
        """
        X_train = self._validate_features(X_train, "X_train")
        y_train = self._validate_labels(y_train, X_train.shape[0])
        self.classes_ = np.array([0, 1])
        self.initialize()
        labels = qml.numpy.array(y_train, requires_grad=False)
        features = [qml.numpy.array(row, requires_grad=False) for row in X_train]
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)
        self.loss_history = []

        def cost_fn(params):
            expectations = qml.numpy.stack([self.qnode(row, params) for row in features])
            probabilities = qml.numpy.clip((expectations + 1.0) / 2.0, 1e-7, 1.0 - 1e-7)
            return -qml.numpy.mean(
                labels * qml.numpy.log(probabilities)
                + (1.0 - labels) * qml.numpy.log(1.0 - probabilities)
            )

        start_time = time.perf_counter()
        final_loss = None
        for _ in range(self.max_iterations):
            self.params, loss = optimizer.step_and_cost(cost_fn, self.params)
            final_loss = float(loss)
            self.loss_history.append(final_loss)
        self.training_time = time.perf_counter() - start_time

        parameter_update_norm = float(np.linalg.norm(np.asarray(self.params) - self.initial_params))
        if not np.isfinite(parameter_update_norm) or parameter_update_norm == 0.0:
            raise ValueError("Optimizer did not produce a finite parameter update")
        logger.info(
            "Quantum training complete: iterations=%d, loss=%.6f, time=%.4fs",
            self.max_iterations,
            final_loss,
            self.training_time,
        )
        return {
            "training_time": self.training_time,
            "final_loss": final_loss,
            "iterations": self.max_iterations,
            "parameter_update_norm": parameter_update_norm,
            "n_records": X_train.shape[0],
            "timing_interval": "optimization_and_simulator_execution",
        }

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        """Execute the trained circuit and return labels plus expectation scores."""
        if self.params is None or self.classes_ is None:
            raise ValueError("Model not yet trained. Call train() first.")
        X_test = self._validate_features(X_test, "X_test")
        start_time = time.perf_counter()
        scores = np.asarray([self.evaluate_circuit(row) for row in X_test], dtype=float)
        probabilities = (scores + 1.0) / 2.0
        y_pred = np.where(scores >= 0.0, 1, 0).astype(int)
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        return y_pred, {
            "scores": scores,
            "probabilities": probabilities,
            "inference_time": inference_time,
            "n_records": X_test.shape[0],
            "timing_interval": "circuit_execution_and_postprocessing",
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
        """Evaluate the fixed test set without changing trainable parameters."""
        X_test = self._validate_features(X_test, "X_test")
        y_test = self._validate_labels(y_test, X_test.shape[0]).astype(int)
        y_pred, details = self.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred, average="binary")
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
            raise ValueError(f"Quantum metrics are not finite and bounded: {metrics}")
        return {
            "metrics": metrics,
            "confusion_matrix": compute_confusion_matrix(y_test, y_pred).tolist(),
            "n_records": X_test.shape[0],
            "inference_time": details["inference_time"],
            "training": self.get_stats(),
        }

    def get_stats(self) -> Dict[str, object]:
        """Return circuit, optimizer, timing, and training metadata."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self.n_params,
            "encoding": self.encoding,
            "simulator": "default.qubit",
            "shots": self.shots,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "random_state": self.random_state,
            "total_training_time": self.training_time,
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0.0,
            "parameter_update_norm": (
                float(np.linalg.norm(np.asarray(self.params) - self.initial_params))
                if self.params is not None and self.initial_params is not None
                else 0.0
            ),
            "final_loss": self.loss_history[-1] if self.loss_history else None,
        }
