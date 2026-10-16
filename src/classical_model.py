"""
Classical baseline model: SGDClassifier with incremental updates.

This module provides:
1. Wrapper around scikit-learn's SGDClassifier
2. Incremental training via partial_fit()
3. Separate timing for training, updates, and inference
4. Reproducible results with fixed random seed
"""

import numpy as np
import time
import logging
from sklearn.linear_model import SGDClassifier
from typing import Tuple, Dict, Optional, List

logger = logging.getLogger(__name__)


class ClassicalModel:
    """
    Classical baseline using SGDClassifier with incremental updates.
    """
    
    def __init__(
        self,
        loss: str = 'log_loss',
        penalty: str = 'l2',
        random_state: int = 42,
        eta0: float = 0.01,
        n_jobs: int = -1
    ):
        """
        Initialize classical model.
        
        Parameters
        ----------
        loss : str, optional
            Loss function (default: 'log_loss')
        penalty : str, optional
            Regularization penalty (default: 'l2')
        random_state : int, optional
            Random seed (default: 42)
        eta0 : float, optional
            Initial learning rate (default: 0.01)
        n_jobs : int, optional
            Number of jobs for parallel processing (default: -1)
        """
        self.loss = loss
        self.penalty = penalty
        self.random_state = random_state
        self.eta0 = eta0
        self.n_jobs = n_jobs
        
        self.model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            random_state=random_state,
            eta0=eta0,
            max_iter=1000,
            tol=1e-3,
            n_jobs=n_jobs,
            warm_start=True,
            verbose=0
        )
        
        self.classes_ = None
        self.training_time = 0.0
        self.update_times = []
        self.inference_times = []
        self.n_batches_seen = 0
        
        logger.info(f"ClassicalModel initialized: loss={loss}, penalty={penalty}, random_state={random_state}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, float]:
        """
        Train model on initial training data.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        
        Returns
        -------
        Dict[str, float]
            Timing information
        """
        if X_train.shape[0] == 0:
            raise ValueError("Empty training set")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatched feature and target lengths")
        
        self.classes_ = np.unique(y_train)
        
        start_time = time.perf_counter()
        self.model.partial_fit(X_train, y_train, classes=self.classes_)
        self.training_time = time.perf_counter() - start_time
        self.n_batches_seen = 1
        
        logger.info(f"Training complete: {self.training_time:.4f} seconds")
        
        return {'training_time': self.training_time}
    
    def update(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray
    ) -> Dict[str, float]:
        """
        Update model with new batch of data (incremental learning).
        
        Parameters
        ----------
        X_batch : np.ndarray
            Batch features
        y_batch : np.ndarray
            Batch targets
        
        Returns
        -------
        Dict[str, float]
            Timing information
        """
        if self.classes_ is None:
            raise ValueError("Model not yet trained. Call train() first.")
        
        start_time = time.perf_counter()
        self.model.partial_fit(X_batch, y_batch, classes=self.classes_)
        update_time = time.perf_counter() - start_time
        
        self.update_times.append(update_time)
        self.n_batches_seen += 1
        self.training_time += update_time
        
        return {'update_time': update_time}
    
    def predict(
        self,
        X_test: np.ndarray,
        return_proba: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict on test data.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test features
        return_proba : bool, optional
            Return probabilities (default: False)
        
        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            Predictions and timing information
        """
        if self.model is None:
            raise ValueError("Model not yet trained")
        
        start_time = time.perf_counter()
        y_pred = self.model.predict(X_test)
        inference_time = time.perf_counter() - start_time
        
        self.inference_times.append(inference_time)
        
        result = y_pred
        if return_proba:
            y_proba = self.model.predict_proba(X_test)
            result = (y_pred, y_proba)
        
        return result, {'inference_time': inference_time}
    
    def get_stats(self) -> Dict:
        """
        Get model statistics.
        
        Returns
        -------
        Dict
            Model statistics
        """
        stats = {
            'n_batches_seen': self.n_batches_seen,
            'total_training_time': self.training_time,
            'avg_update_time': np.mean(self.update_times) if self.update_times else 0.0,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0,
        }
        return stats


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    from data_processing import load_dataset, split_dataset, preprocess_classical
    
    X, y = load_dataset()
    X_train, _, X_test, y_train, _, y_test = split_dataset(X, y)
    X_train, _, X_test, _ = preprocess_classical(X_train, X_train, X_test)
    
    model = ClassicalModel()
    model.train(X_train[:100], y_train[:100])
    y_pred, timing = model.predict(X_test)
    
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Prediction stats: {model.get_stats()}")
    print("Classical model smoke test passed.")
