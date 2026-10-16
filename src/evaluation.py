"""
Evaluation: Metrics computation, timing analysis, and benchmarking.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    average : str, optional
        Average strategy for multiclass (default: 'binary')
    
    Returns
    -------
    Dict[str, float]
        Metrics dictionary
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Mismatched lengths: {len(y_true)} vs {len(y_pred)}")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    
    Returns
    -------
    np.ndarray
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def compute_throughput(
    n_records: int,
    total_time: float
) -> float:
    """
    Compute throughput in records per second.
    
    Parameters
    ----------
    n_records : int
        Number of records processed
    total_time : float
        Total time in seconds
    
    Returns
    -------
    float
        Throughput in records/second
    """
    if total_time <= 0:
        return 0.0
    return n_records / total_time


def compute_aggregate_metrics(
    individual_results: list
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple runs.
    
    Parameters
    ----------
    individual_results : list
        List of result dictionaries from individual runs
    
    Returns
    -------
    Dict[str, float]
        Aggregated metrics (mean, std)
    """
    if not individual_results:
        raise ValueError("No results to aggregate")
    
    agg_metrics = {}
    
    # Get all metric keys from first result
    for key in individual_results[0].keys():
        values = [r[key] for r in individual_results if isinstance(r[key], (int, float))]
        
        if values:
            agg_metrics[f'{key}_mean'] = np.mean(values)
            agg_metrics[f'{key}_std'] = np.std(values)
    
    return agg_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)
    tp = compute_throughput(100, 2.5)
    
    print(f"Metrics: {metrics}")
    print(f"Confusion matrix:\n{cm}")
    print(f"Throughput: {tp:.2f} rec/sec")
    print("Evaluation smoke test passed.")
