"""Experiment metrics and timing helpers."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_classification_metrics(y_true, y_pred, average="binary"):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    values = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite metric")
    return values


def compute_confusion_matrix(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_throughput(n_records, total_time):
    if n_records < 0 or total_time < 0:
        raise ValueError("counts and times must be non-negative")
    return float(n_records / total_time) if total_time > 0 else 0.0


def compute_aggregate_metrics(individual_results):
    if not individual_results:
        raise ValueError("No results to aggregate")
    keys = set.intersection(*(set(r) for r in individual_results))
    out = {}
    for key in sorted(keys):
        vals = [r[key] for r in individual_results if isinstance(r[key], (int, float, np.number)) and np.isfinite(r[key])]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    return out
