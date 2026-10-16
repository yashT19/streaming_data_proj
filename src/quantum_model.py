"""
Quantum model: PennyLane-based variational quantum circuit classifier.

This module provides:
1. Quantum circuit definition with angle encoding
2. Parameterized trainable gates and entanglement
3. Classical optimizer for training
4. Separate training and inference timing
5. Reproducible results with fixed random seed
"""

import numpy as np
import time
import logging
from typing import Tuple, Dict, Optional
import warnings

logger = logging.getLogger(__name__)

# Try to import PennyLane; provide graceful fallback
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not installed. Quantum models will not function. "
                   "Install with: pip install pennylane")


class QuantumModel:
    """
    Quantum classifier using PennyLane's default.qubit simulator.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = 'angle',
        optimizer: str = 'adam',
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        random_state: int = 42,
        shots: Optional[int] = None,
    ):
        """
        Initialize quantum model.
        
        Parameters
        ----------
        n_qubits : int, optional
            Number of qubits (default: 4)
        n_layers : int, optional
            Number of circuit layers (default: 2)
        encoding : str, optional
            Encoding strategy: 'angle' (default)
        optimizer : str, optional
            Optimizer: 'adam' (default)
        learning_rate : float, optional
            Learning rate (default: 0.01)
        max_iterations : int, optional
            Maximum optimization iterations (default: 100)
        random_state : int, optional
            Random seed (default: 42)
        shots : Optional[int], optional
            Number of measurement shots (None = statevector, default: None)
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is not installed. Install with: pip install pennylane")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.shots = shots
        
        np.random.seed(random_state)
        
        # Initialize device
        if shots is None:
            self.dev = qml.device('default.qubit', wires=n_qubits)
        else:
            self.dev = qml.device('default.qubit', wires=n_qubits, shots=shots)
        
        self.params = None
        self.opt = None
        self.classes_ = None
        self.training_time = 0.0
        self.inference_times = []
        
        logger.info(f"QuantumModel initialized: n_qubits={n_qubits}, n_layers={n_layers}, "
                   f"optimizer={optimizer}, shots={shots}")
    
    def _circuit(self, inputs: np.ndarray, params: np.ndarray) -> float:
        """
        Define quantum circuit with angle encoding and parameterized gates.
        
        Parameters
        ----------
        inputs : np.ndarray
            Input features (should be n_qubits features)
        params : np.ndarray
            Parameterized gate angles
        
        Returns
        -------
        float
            Measurement expectation value on first qubit
        """
        # Angle encoding
        for i, x in enumerate(inputs[:self.n_qubits]):
            qml.RX(x, wires=i)
        
        # Parameterized layers with entanglement
        param_idx = 0
        for layer in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(params[param_idx], wires=i)
                param_idx += 1
            
            # Entangling layer (CNOT ring)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        
        # Final measurement
        return qml.expval(qml.PauliZ(0))
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train quantum model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features (should be n_qubits features)
        y_train : np.ndarray
            Training targets (binary: 0 or 1)
        
        Returns
        -------
        Dict[str, float]
            Timing information
        """
        if X_train.shape[0] == 0:
            raise ValueError("Empty training set")
        if X_train.shape[1] < self.n_qubits:
            raise ValueError(f"X_train has {X_train.shape[1]} features, need {self.n_qubits}")
        
        self.classes_ = np.unique(y_train)
        if len(self.classes_) != 2:
            raise ValueError(f"Binary classification required, got {len(self.classes_)} classes")
        
        # Initialize parameters
        n_params = self.n_layers * self.n_qubits  # One RY per qubit per layer
        self.params = np.random.randn(n_params) * 0.1
        
        # Create QNode
        @qml.qnode(self.dev)
        def cost_fn(params):
            predictions = []
            for x in X_train:
                pred = self._circuit(x, params)
                predictions.append(pred)
            predictions = np.array(predictions)
            
            # Convert to binary predictions
            y_pred = (predictions > 0).astype(int)
            
            # Binary cross-entropy loss
            eps = 1e-7
            loss = -np.mean(
                y_train * np.log(np.clip((predictions + 1) / 2, eps, 1 - eps)) +
                (1 - y_train) * np.log(1 - np.clip((predictions + 1) / 2, eps, 1 - eps))
            )
            return loss
        
        # Optimize
        if self.optimizer_name == 'adam':
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        else:
            opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        start_time = time.perf_counter()
        
        for iteration in range(self.max_iterations):
            self.params, loss = opt.step_and_cost(cost_fn, self.params)
            
            if iteration % max(1, self.max_iterations // 5) == 0:
                logger.debug(f"Iteration {iteration}: loss = {loss:.4f}")
        
        self.training_time = time.perf_counter() - start_time
        logger.info(f"Training complete: {self.training_time:.4f} seconds, "
                   f"final loss = {loss:.4f}")
        
        return {'training_time': self.training_time}
    
    def predict(
        self,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict on test data.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test features (should be n_qubits features)
        
        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            Predictions and timing information
        """
        if self.params is None:
            raise ValueError("Model not yet trained. Call train() first.")
        
        @qml.qnode(self.dev)
        def circuit_predict(x):
            return self._circuit(x, self.params)
        
        start_time = time.perf_counter()
        
        predictions = []
        for x in X_test:
            pred = circuit_predict(x)
            # Convert to binary: positive → class 1, negative → class 0
            binary_pred = self.classes_[1] if pred > 0 else self.classes_[0]
            predictions.append(binary_pred)
        
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        
        y_pred = np.array(predictions)
        
        return y_pred, {'inference_time': inference_time}
    
    def get_stats(self) -> Dict:
        """
        Get model statistics.
        
        Returns
        -------
        Dict
            Model statistics
        """
        stats = {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_params': self.n_layers * self.n_qubits,
            'total_training_time': self.training_time,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0,
        }
        return stats


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    
    if PENNYLANE_AVAILABLE:
        from data_processing import load_dataset, split_dataset, preprocess_quantum
        
        X, y = load_dataset()
        X_train, _, X_test, y_train, _, y_test = split_dataset(X, y)
        X_train, _, X_test, _, _ = preprocess_quantum(X_train, X_train, X_test)
        
        model = QuantumModel(n_qubits=4, n_layers=1, max_iterations=10)
        model.train(X_train[:20], y_train[:20])
        y_pred, timing = model.predict(X_test[:5])
        
        print(f"Predictions: {y_pred}")
        print(f"Prediction stats: {model.get_stats()}")
        print("Quantum model smoke test passed.")
    else:
        print("PennyLane not available. Install with: pip install pennylane")
