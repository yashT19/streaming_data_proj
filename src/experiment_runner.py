"""Reproducible controlled comparison for the existing classical and quantum models.

Run with: python -m src.experiment_runner [--smoke]
"""
import argparse
import json
import logging
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from src.data_processing import prepare_data
from src.stream_simulator import StreamSimulator
from src.classical_model import ClassicalModel
from src.quantum_model import QuantumModel
from src.evaluation import compute_classification_metrics, compute_confusion_matrix

LOG = logging.getLogger(__name__)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _record(exp, model, data, seed, batch, length, rep, train_time, update_times,
            predict_time, e2e, y_true, y_pred, n_pred, config, status="success", error=None):
    metrics = compute_classification_metrics(y_true, y_pred) if status == "success" else {}
    cm = compute_confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist() if status == "success" else None
    return {
        "experiment_id": exp, "model_name": model, "split_seed": seed,
        "batch_size": batch, "stream_length": length, "timing_repetition": rep,
        "feature_count": 4, "qubit_count": 4 if model == "quantum" else None,
        "training_records": min(length, len(data["y_train"])), "test_records": len(y_true),
        **{f"metric_{k}": v for k, v in metrics.items()}, "confusion_matrix": json.dumps(cm),
        "model_training_time_seconds": train_time,
        "batch_update_time_seconds": float(sum(update_times)) if update_times else 0.0,
        "batch_update_mean_seconds": float(np.mean(update_times)) if update_times else 0.0,
        "batch_update_count": len(update_times), "batch_update_times_seconds": json.dumps([float(x) for x in update_times]),
        "prediction_time_seconds": predict_time,
        "end_to_end_time_seconds": e2e,
        "throughput_records_per_second": (n_pred / e2e if e2e > 0 else 0.0),
        "throughput_numerator_records": n_pred,
        "throughput_interval": "training_plus_stream_prediction_plus_fixed_test_prediction",
        "training_protocol": "partial_fit_each_stream_batch" if model == "classical" else "initial_window_train_then_prediction_only",
        "framework": "scikit-learn" if model == "classical" else "PennyLane",
        "simulator": "none" if model == "classical" else "default.qubit",
        "optimizer": None if model == "classical" else config["quantum_model"]["optimizer"],
        "optimizer_iterations": None if model == "classical" else config["quantum_model"]["max_iterations"],
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__, "status": status,
        "error_type": type(error).__name__ if error else None, "error_message": str(error) if error else None,
    }


def _run_one(model_name, data, seed, batch_size, stream_length, rep, cfg, exp):
    X = data["X_train_quantum"]
    y = data["y_train"]
    n = min(stream_length, len(y))
    X, y = X[:n], y[:n]
    Xtest, yt = data["X_test_quantum"], data["y_test"]
    start = time.perf_counter()
    updates, train_time, predtime, n_pred = [], 0.0, 0.0, 0
    try:
        if model_name == "classical":
            model = ClassicalModel(random_state=seed)
            for xb, yb, _ in StreamSimulator(X, y, batch_size=batch_size, random_state=seed).stream_batches():
                model.process_batch(xb, yb, classes=np.unique(y) if not updates else None)
                updates.append(model.update_times[-1])
                train_time = float(sum(model.update_times))
                # Online prediction of each batch after the update is timed as part of end-to-end protocol.
                t1 = time.perf_counter(); model.predict(xb); predtime += time.perf_counter() - t1; n_pred += len(yb)
            t1 = time.perf_counter(); yp, _ = model.predict(Xtest); test_pred = time.perf_counter() - t1
            predtime += test_pred
            e2e = time.perf_counter() - start
        else:
            qc = cfg["quantum_model"]
            model = QuantumModel(n_qubits=4, n_layers=qc["n_layers"], learning_rate=qc["learning_rate"],
                                 max_iterations=qc["max_iterations"], random_state=seed, shots=qc.get("shots"), max_input_abs=100.0)
            # Documented protocol: optimize on first stream window, then predict only on the rest.
            window = min(batch_size, len(y))
            t0 = time.perf_counter(); model.train(X[:window], y[:window]); train_time = time.perf_counter() - t0
            rest = X[window:]
            for i in range(0, len(rest), batch_size):
                xb = rest[i:i + batch_size]
                t1 = time.perf_counter(); model.predict(xb); predtime += time.perf_counter() - t1; n_pred += len(xb)
            t1 = time.perf_counter(); yp, _ = model.predict(Xtest); predtime += time.perf_counter() - t1
            e2e = time.perf_counter() - start
            n_pred += len(Xtest)
            # Test scores are always from the fixed held-out test set.
            yp, _ = model.predict(Xtest)
        return _record(exp, model_name, data, seed, batch_size, n, rep, train_time, updates,
                       predtime, e2e, yt, yp, n_pred, cfg)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return _record(exp, model_name, data, seed, batch_size, n, rep, train_time, updates,
                       predtime, elapsed, yt, np.zeros_like(yt), n_pred, cfg, "failed", exc)


def run_experiments(config_path="configs/experiment.yaml", smoke=False):
    cfg = load_config(config_path)
    qcfg = cfg["quantum_model"].copy()
    if smoke:
        qcfg["max_iterations"] = min(2, qcfg["max_iterations"])
        cfg["quantum_model"] = qcfg
    seed = int(cfg["dataset"]["random_state"])
    data = prepare_data(random_state=seed, n_quantum_features=4, verbose=False)
    assert data["X_train_quantum"].shape[1] == data["X_test_quantum"].shape[1] == 4
    assert len(data["y_test"]) == 114
    lengths = [min(int(v), len(data["y_train"])) for v in cfg["streaming"].get("stream_lengths", [100, 200, 341])]
    lengths = sorted(set(lengths + [len(data["y_train"])]))
    batches = cfg["streaming"].get("batch_sizes", [16, 32, 64])
    reps = 1 if smoke else 3
    jobs = [("A_performance", 32, len(data["y_train"]))]
    if not smoke:
        jobs += [("B_batch_sensitivity", b, len(data["y_train"])) for b in batches]
        jobs += [("C_stream_sensitivity", 32, n) for n in lengths]
    rows = []
    for exp, batch, length in jobs:
        for model in ("classical", "quantum"):
            for rep in range(reps):
                rows.append(_run_one(model, data, seed, batch, length, rep, cfg, exp))
    out = Path(cfg.get("output", {}).get("results_dir", "results")); out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / ("smoke_raw_results.csv" if smoke else "raw_results.csv"), index=False)
    successful = raw[raw.status == "success"]
    keys = ["experiment_id", "model_name", "batch_size", "stream_length"]
    metrics = [c for c in raw.columns if c.startswith("metric_") and c != "metric_undefined"]
    numeric = metrics + ["model_training_time_seconds", "batch_update_time_seconds", "prediction_time_seconds", "end_to_end_time_seconds", "throughput_records_per_second"]
    summary = successful.groupby(keys)[numeric].agg(["mean", "std"]).reset_index()
    summary.to_csv(out / ("smoke_summary.csv" if smoke else "summary.csv"), index=False)
    # Integrity checks are part of the runner's success condition.
    if raw.empty or (raw.status == "failed").any():
        raise RuntimeError(f"Experiment failures recorded: {raw[raw.status == 'failed'][['experiment_id','model_name','error_message']].to_dict('records')}")
    assert np.isfinite(successful[numeric].to_numpy(dtype=float)).all()
    assert (successful[["model_training_time_seconds", "batch_update_time_seconds", "prediction_time_seconds", "end_to_end_time_seconds"]] >= 0).all().all()
    assert (successful.test_records == len(data["y_test"])).all()
    pd.DataFrame([{"mode": "smoke" if smoke else "required", "split_seed": seed,
                   "test_records": len(data["y_test"]), "feature_count": 4,
                   "quantum_iterations": qcfg["max_iterations"], "timing_repetitions": reps}]).to_json(out / "experiment_metadata.json", orient="records", indent=2)
    LOG.info("Saved %d observations under %s", len(raw), out)
    return raw, summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default="configs/experiment.yaml"); args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiments(args.config, args.smoke)


if __name__ == "__main__":
    main()




