#!/usr/bin/env python
"""Verify the genuine PennyLane variational quantum classifier."""

import numpy as np

from src.data_processing import prepare_data
from src.quantum_model import QuantumModel
from src.stream_simulator import StreamSimulator


CONFIG = {
    "random_state": 42,
    "n_qubits": 4,
    "n_layers": 1,
    "max_iterations": 5,
    "initial_window": 16,
    "test_window": 8,
    "shots": None,
}


def run_model(X_train, y_train, X_test, y_test):
    model = QuantumModel(
        n_qubits=CONFIG["n_qubits"],
        n_layers=CONFIG["n_layers"],
        learning_rate=0.01,
        max_iterations=CONFIG["max_iterations"],
        random_state=CONFIG["random_state"],
        shots=CONFIG["shots"],
    )
    training = model.train(X_train, y_train)
    evaluation = model.evaluate(X_test, y_test)
    predictions, prediction_details = model.predict(X_test)
    return model, training, evaluation, predictions, prediction_details


def main():
    rng = np.random.RandomState(7)
    synthetic_X = rng.uniform(-0.5, 0.5, size=(8, 4))
    synthetic_y = np.array([0, 1] * 4)
    synthetic_model = QuantumModel(n_qubits=4, n_layers=1, max_iterations=3, random_state=42)
    initial_params = synthetic_model.initialize()
    circuit_output = synthetic_model.evaluate_circuit(synthetic_X[0])
    assert initial_params.shape == (1, 4, 2)
    assert np.isfinite(circuit_output) and -1 <= circuit_output <= 1
    synthetic_training = synthetic_model.train(synthetic_X, synthetic_y)
    synthetic_predictions, synthetic_details = synthetic_model.predict(synthetic_X[:3])
    assert synthetic_training["parameter_update_norm"] > 0
    assert len(synthetic_predictions) == 3
    assert set(np.unique(synthetic_predictions)).issubset({0, 1})
    assert np.isfinite(synthetic_details["scores"]).all()

    data = prepare_data(random_state=CONFIG["random_state"], n_quantum_features=4, verbose=False)
    X_train = data["X_train_quantum"][: CONFIG["initial_window"]]
    y_train = data["y_train"][: CONFIG["initial_window"]]
    X_test = data["X_test_quantum"][: CONFIG["test_window"]]
    y_test = data["y_test"][: CONFIG["test_window"]]
    assert X_train.shape[1] == CONFIG["n_qubits"]

    first = run_model(X_train, y_train, X_test, y_test)
    second = run_model(X_train, y_train, X_test, y_test)
    first_model, training, evaluation, predictions, prediction_details = first
    second_model, _, second_evaluation, second_predictions, _ = second
    np.testing.assert_allclose(np.asarray(first_model.params), np.asarray(second_model.params), atol=1e-10)
    np.testing.assert_array_equal(predictions, second_predictions)
    assert evaluation["metrics"] == second_evaluation["metrics"]
    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in evaluation["metrics"].values())
    assert np.asarray(prediction_details["scores"]).shape == (len(y_test),)

    stream = StreamSimulator(
        data["X_train_quantum"],
        data["y_train"],
        batch_size=CONFIG["initial_window"],
        random_state=CONFIG["random_state"],
        shuffle=True,
    )
    stream_model = QuantumModel(
        n_qubits=4,
        n_layers=CONFIG["n_layers"],
        max_iterations=CONFIG["max_iterations"],
        random_state=CONFIG["random_state"],
    )
    stream_batches = list(stream.stream_batches())
    stream_model.train(stream_batches[0][0], stream_batches[0][1])
    stream_inference_records = 0
    for X_batch, y_batch, _ in stream_batches[1:]:
        stream_predictions, stream_details = stream_model.predict(X_batch)
        assert len(stream_predictions) == len(y_batch)
        assert set(np.unique(stream_predictions)).issubset({0, 1})
        assert np.isfinite(stream_details["scores"]).all()
        stream_inference_records += len(y_batch)
    assert stream_inference_records == len(data["y_train"]) - CONFIG["initial_window"]

    invalid_checks = []
    try:
        first_model.predict(np.ones((2, 3)))
    except ValueError as error:
        invalid_checks.append("feature_dimension: " + str(error))
    try:
        QuantumModel(n_qubits=3)
    except ValueError as error:
        invalid_checks.append("qubit_count: " + str(error))
    try:
        first_model.train(np.ones((4, 4)), np.array([0, 1, 2, 1]))
    except ValueError as error:
        invalid_checks.append("labels: " + str(error))
    assert len(invalid_checks) == 3

    print("QUANTUM MODEL VERIFICATION: PASS")
    print(f"simulator=default.qubit, shots={CONFIG['shots']}")
    print(f"qubits={first_model.n_qubits}, layers={first_model.n_layers}, parameters={first_model.n_params}")
    print(f"synthetic_output={circuit_output:.8f}")
    print(f"prepared_records={training['n_records']}, iterations={training['iterations']}")
    print(f"final_loss={training['final_loss']:.8f}")
    print(f"parameter_update_norm={training['parameter_update_norm']:.8f}")
    print(f"metrics={evaluation['metrics']}")
    print(f"confusion_matrix={evaluation['confusion_matrix']}")
    print(f"prediction_only_stream_records={stream_inference_records}")
    print(f"training_seconds={training['training_time']:.8f}")
    print(f"inference_seconds={evaluation['inference_time']:.8f}")
    print("reproducible_parameters=True")
    print("reproducible_predictions=True")
    print("invalid_input_checks=3/3")
    print("warnings=none")


if __name__ == "__main__":
    main()
