"""Tests for the incremental classical baseline."""

import numpy as np
import pytest

from src.classical_model import ClassicalModel


def test_partial_fit_stream_and_evaluation():
    rng = np.random.RandomState(11)
    X = rng.randn(30, 4)
    y = np.array([0] * 10 + [1] * 20)
    model = ClassicalModel(random_state=42, n_jobs=1)

    model.process_batch(X[:5], y[:5], classes=[0, 1])
    model.process_batch(X[5:20], y[5:20])
    result = model.evaluate(X[20:], y[20:])

    assert model.get_stats()["n_batches_seen"] == 2
    assert model.get_stats()["n_records_processed"] == 20
    assert result["n_records"] == 10
    assert np.asarray(result["confusion_matrix"]).shape == (2, 2)
    assert all(0.0 <= value <= 1.0 for value in result["metrics"].values())
    assert all(np.isfinite(value) for value in result["metrics"].values())


def test_first_batch_requires_complete_class_list():
    model = ClassicalModel(n_jobs=1)
    X = np.ones((2, 4))
    y = np.zeros(2, dtype=int)

    with pytest.raises(ValueError, match="complete classes"):
        model.process_batch(X, y)


def test_invalid_inputs_have_useful_errors():
    model = ClassicalModel(n_jobs=1)
    with pytest.raises(ValueError, match="2D"):
        model.process_batch(np.ones(4), np.array([0, 1]), classes=[0, 1])

    model.process_batch(np.ones((2, 4)), np.array([0, 1]), classes=[0, 1])
    with pytest.raises(ValueError, match="matching lengths"):
        model.evaluate(np.ones((3, 4)), np.array([0, 1]))


def test_prediction_scores_are_available():
    model = ClassicalModel(n_jobs=1)
    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.array([0, 1] * 5)
    model.process_batch(X, y, classes=[0, 1])

    predictions, details = model.predict(X, return_scores=True)
    assert predictions.shape == (10,)
    assert details["scores"].shape == (10,)
    assert details["n_records"] == 10
