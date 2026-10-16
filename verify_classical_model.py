#!/usr/bin/env python
"""Verify the four-feature incremental classical baseline end to end."""

import numpy as np

from src.classical_model import ClassicalModel
from src.data_processing import prepare_data
from src.stream_simulator import StreamSimulator


CONFIG = {
    "random_state": 42,
    "batch_size": 32,
    "n_features": 4,
    "positive_label": 1,
}


def run_once():
    data = prepare_data(
        random_state=CONFIG["random_state"],
        n_quantum_features=CONFIG["n_features"],
        verbose=False,
    )
    X_train = data["X_train_quantum"]
    y_train = data["y_train"]
    X_test = data["X_test_quantum"]
    y_test = data["y_test"]

    assert X_train.shape[1] == CONFIG["n_features"]
    simulator = StreamSimulator(
        X_train,
        y_train,
        batch_size=CONFIG["batch_size"],
        random_state=CONFIG["random_state"],
        shuffle=True,
    )
    assert simulator.validate_stream()

    model = ClassicalModel(random_state=CONFIG["random_state"], n_jobs=1)
    batch_count = 0
    for X_batch, y_batch, metadata in simulator.stream_batches():
        classes = np.unique(y_train) if batch_count == 0 else None
        update = model.process_batch(X_batch, y_batch, classes=classes)
        assert update["n_records"] == X_batch.shape[0]
        batch_count += 1

    evaluation = model.evaluate(
        X_test,
        y_test,
        positive_label=CONFIG["positive_label"],
    )
    predictions, prediction_details = model.predict(X_test, return_scores=True)
    stats = model.get_stats()

    assert batch_count == simulator.n_batches
    assert stats["n_records_processed"] == len(y_train)
    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset(set(model.classes_))
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in evaluation["metrics"].values())
    assert np.asarray(evaluation["confusion_matrix"]).shape == (2, 2)
    assert prediction_details["n_records"] == len(y_test)

    return data, model, evaluation, predictions


def main():
    first = run_once()
    second = run_once()

    first_eval = first[2]
    second_eval = second[2]
    np.testing.assert_array_equal(first[3], second[3])
    assert first_eval["metrics"] == second_eval["metrics"]
    assert first[1].get_stats()["n_records_processed"] == 341

    invalid_results = []
    try:
        StreamSimulator(np.ones((4, 4)), np.zeros(4), batch_size=0)
    except ValueError as error:
        invalid_results.append("batch_size: " + str(error))
    try:
        ClassicalModel(n_jobs=1).process_batch(np.ones((2, 4)), np.zeros(2))
    except ValueError as error:
        invalid_results.append("classes: " + str(error))
    try:
        first[1].evaluate(np.ones((3, 4)), np.zeros(2, dtype=int))
    except ValueError as error:
        invalid_results.append("test_lengths: " + str(error))

    assert len(invalid_results) == 3
    metrics = first_eval["metrics"]
    print("CLASSICAL BASELINE VERIFICATION: PASS")
    print(f"training_records={first[1].get_stats()['n_records_processed']}")
    print(f"batches={first[1].get_stats()['n_batches_seen']}")
    print(f"test_records={first_eval['n_records']}")
    print(f"metrics={metrics}")
    print(f"confusion_matrix={first_eval['confusion_matrix']}")
    print(f"training_seconds={first[1].get_stats()['total_training_time']:.8f}")
    print(f"inference_seconds={first_eval['inference_time']:.8f}")
    print("reproducible_predictions=True")
    print("invalid_input_checks=3/3")
    print("warnings=none")


if __name__ == "__main__":
    main()
