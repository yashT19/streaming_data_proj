"""Incremental classical baseline for the streaming experiment.

The baseline uses scikit-learn's ``SGDClassifier`` with log loss and L2
regularization.  It is intentionally fed one simulated stream batch at a
time through ``partial_fit``; the fixed class list is supplied on the first
update so a first batch containing only one class is still valid.
"""

import numpy as np
import time
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


class ClassicalModel:
    """An ``SGDClassifier`` wrapper with explicit stream and evaluation state."""
    
    def __init__(
        self,
        loss: str = 'log_loss',
        penalty: str = 'l2',
        random_state: int = 42,
        eta0: float = 0.01,
        n_jobs: int = -1,
        max_iter: int = 1000,
        tol: Optional[float] = 1e-3,
        l1_ratio: float = 0.15,
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
        max_iter : int, optional
            Maximum iterations used by each ``partial_fit`` update.
        tol : float or None, optional
            Stopping tolerance passed to ``SGDClassifier``.
        l1_ratio : float, optional
            Elastic-net mixing value; retained in the configuration even
            though the default L2 penalty does not use it.
        """
        self.loss = loss
        self.penalty = penalty
        self.random_state = random_state
        self.eta0 = eta0
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        
        self.model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            random_state=random_state,
            eta0=eta0,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            warm_start=True,
            l1_ratio=l1_ratio,
            verbose=0
        )
        
        self.classes_ = None
        self.training_time = 0.0
        self.update_times = []
        self.inference_times = []
        self.update_records = []
        self.n_batches_seen = 0
        
        logger.info(f"ClassicalModel initialized: loss={loss}, penalty={penalty}, random_state={random_state}")
    
    @staticmethod
    def _validate_batch(X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_batch = np.asarray(X_batch)
        y_batch = np.asarray(y_batch).reshape(-1)
        if X_batch.ndim != 2:
            raise ValueError(f"Features must be a 2D array, got shape {X_batch.shape}")
        if X_batch.shape[0] == 0:
            raise ValueError("Training batch must not be empty")
        if X_batch.shape[0] != y_batch.shape[0]:
            raise ValueError(
                f"Mismatched batch lengths: {X_batch.shape[0]} features vs {y_batch.shape[0]} labels"
            )
        if not np.isfinite(X_batch).all():
            raise ValueError("Training features contain NaN or infinite values")
        return X_batch, y_batch

    def initialize(self, classes: Iterable[int]) -> None:
        """Set the complete class list required by the first update."""
        classes_array = np.asarray(list(classes)).reshape(-1)
        if classes_array.size < 2 or np.unique(classes_array).size != classes_array.size:
            raise ValueError("classes must contain at least two unique labels")
        self.classes_ = np.sort(classes_array)

    def process_batch(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        classes: Optional[Iterable[int]] = None,
    ) -> Dict[str, float]:
        """
        Apply exactly one sequential update with ``partial_fit``.
        
        Parameters
        ----------
        X_batch : np.ndarray
            Features for one stream batch.
        y_batch : np.ndarray
            Labels for one stream batch.
        classes : iterable, optional
            Complete class list, required on the first call. Passing it on
            later calls is allowed and is checked against the initialized list.
        
        Returns
        -------
        Dict[str, float]
            Timing information
        """
        X_batch, y_batch = self._validate_batch(X_batch, y_batch)
        if self.classes_ is None:
            if classes is None:
                raise ValueError("The complete classes list is required on the first batch")
            self.initialize(classes)
        elif classes is not None and not np.array_equal(self.classes_, np.sort(np.asarray(list(classes)))):
            raise ValueError("classes does not match the initialized class list")

        start_time = time.perf_counter()
        self.model.partial_fit(X_batch, y_batch, classes=self.classes_)
        update_time = time.perf_counter() - start_time
        self.update_times.append(update_time)
        self.update_records.append(X_batch.shape[0])
        self.training_time += update_time
        self.n_batches_seen += 1

        return {
            'update_time': update_time,
            'cumulative_training_time': self.training_time,
            'n_records': X_batch.shape[0],
            'cumulative_records': int(sum(self.update_records)),
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train one initial batch, using all labels to define the classes."""
        return self.process_batch(X_train, y_train, classes=np.unique(y_train))
    
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
        return self.process_batch(X_batch, y_batch)
    
    def predict(
        self,
        X_test: np.ndarray,
        return_proba: bool = False,
        return_scores: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
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
        if self.classes_ is None:
            raise ValueError("Model not yet trained")
        X_test = np.asarray(X_test)
        if X_test.ndim != 2:
            raise ValueError(f"Features must be a 2D array, got shape {X_test.shape}")
        if not np.isfinite(X_test).all():
            raise ValueError("Prediction features contain NaN or infinite values")
        
        start_time = time.perf_counter()
        y_pred = self.model.predict(X_test)
        inference_time = time.perf_counter() - start_time
        
        self.inference_times.append(inference_time)
        
        result = y_pred
        details: Dict[str, object] = {
            'inference_time': inference_time,
            'n_records': X_test.shape[0],
            'timing_interval': 'predict_only',
        }
        if return_proba:
            y_proba = self.model.predict_proba(X_test)
            details['probabilities'] = y_proba
        if return_scores:
            details['scores'] = self.model.decision_function(X_test)
        
        return result, details

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        positive_label: int = 1,
    ) -> Dict[str, object]:
        """Evaluate on a fixed held-out set without changing model state."""
        y_test = np.asarray(y_test).reshape(-1)
        if y_test.size == 0:
            raise ValueError("Test labels must not be empty")
        if np.asarray(X_test).shape[0] != y_test.size:
            raise ValueError("Test features and labels must have matching lengths")
        y_pred, timing = self.predict(X_test)
        if self.classes_.size == 2:
            if positive_label not in self.classes_:
                raise ValueError(f"positive_label {positive_label} is not in model classes {self.classes_.tolist()}")
            average = 'binary'
            metric_kwargs = {'pos_label': positive_label}
        else:
            average = 'macro'
            metric_kwargs = {}

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average, zero_division=0, **metric_kwargs),
            'recall': recall_score(y_test, y_pred, average=average, zero_division=0, **metric_kwargs),
            'f1': f1_score(y_test, y_pred, average=average, zero_division=0, **metric_kwargs),
        }
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
            raise ValueError(f"Classification metrics are not finite and bounded: {metrics}")

        return {
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=self.classes_).tolist(),
            'positive_label': positive_label if self.classes_.size == 2 else None,
            'n_records': int(y_test.size),
            'inference_time': timing['inference_time'],
            'timing_interval': 'fixed_test_prediction',
            'training': self.get_stats(),
        }
    
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
            'n_records_processed': int(sum(self.update_records)),
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'configuration': {
                'loss': self.loss,
                'penalty': self.penalty,
                'random_state': self.random_state,
                'eta0': self.eta0,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'l1_ratio': self.l1_ratio,
            },
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
