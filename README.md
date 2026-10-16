# Quantum vs Classical Approaches for Streaming Data Analytics: An Experimental Study

An academic research project comparing classical and quantum machine learning approaches for simulated streaming data classification.

## Research Questions

1. How do predictive performance metrics compare?
2. How do training, updating, and inference costs compare?
3. How do batch size and stream length affect performance?
4. What limitations prevent us from concluding that a quantum approach is universally better?
5. How do the models behave under simulated concept drift (optional)?

## Project Structure

```
quantum-streaming-study/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project metadata
├── .gitignore                  # Git ignore rules
├── configs/
│   └── experiment.yaml         # Experiment configuration
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Dataset loading and preprocessing
│   ├── stream_simulator.py     # Streaming data simulator
│   ├── classical_model.py      # Classical baseline (SGDClassifier)
│   ├── quantum_model.py        # Quantum circuit classifier
│   ├── evaluation.py           # Metrics and evaluation
│   ├── experiment_runner.py    # Experiment orchestration
│   ├── plotting.py             # Result visualization
│   └── reproducibility.py      # Reproducibility utilities
├── tests/
│   ├── test_data_processing.py
│   ├── test_stream_simulator.py
│   ├── test_classical_model.py
│   ├── test_quantum_model.py
│   ├── test_evaluation.py
│   └── test_experiment_runner.py
├── notebooks/
│   └── exploration.ipynb       # Optional exploration notebook
├── results/
│   └── .gitkeep
└── docs/
    ├── IMPLEMENTATION_PLAN.md  # Detailed implementation plan
    ├── METHODOLOGY.md          # Experimental methodology
    ├── RESULTS.md              # Experimental results
    └── LIMITATIONS.md          # Known limitations
```

## Prerequisites

- Python 3.10 or later
- pip or conda
- Git

## Installation

### Create a virtual environment

```bash
python -m venv venv
source venv/Scripts/Activate.ps1  # Windows PowerShell
# or
venv\Scripts\activate.bat         # Windows CMD
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import pennylane; import numpy; import sklearn; print('All core dependencies OK')"
```

## Quick Start

### Run the smoke test (quick validation)
```bash
pytest tests/ -v -k "smoke" --tb=short
```

### Run the full unit test suite
```bash
pytest tests/ -v
```

### Run a small end-to-end experiment
```bash
python -m src.experiment_runner --config configs/experiment.yaml --mode smoke
```

### Verify the classical streaming baseline
```bash
python verify_classical_model.py
```

This verification uses the prepared four-feature representation, streams all
341 training records through `SGDClassifier.partial_fit()`, evaluates once on
the fixed 114-record test set, repeats the run for reproducibility, and checks
invalid inputs. The baseline uses `log_loss`, L2 regularization, `eta0=0.01`,
and `random_state=42`; class `1` is the binary positive label.

### Verify the variational quantum classifier
```bash
python verify_quantum_model.py
```

This runs a genuine PennyLane `default.qubit` circuit on synthetic data and a
small initial window of the prepared four-feature data. The quantum model is
trained once on that initial window; it does not implement `partial_fit()`.

### Run the full experiment (may take several minutes)
```bash
python -m src.experiment_runner --config configs/experiment.yaml --mode full
```

### Generate plots from results
```bash
python -c "from src.plotting import generate_plots; generate_plots('results/')"
```

## Expected Output

The project generates:

- **CSV/JSON results** in `results/` containing raw experiment metrics
- **Summary tables** with aggregated performance metrics
- **Plots** showing accuracy, timing, throughput, and other comparisons
- **Test reports** from pytest

## Framework Selection

This project uses **PennyLane** with a NumPy-based simulator for quantum circuits. This choice prioritizes:

- Ease of installation and compatibility
- CPU-only execution without external dependencies
- Clear circuit abstraction and parameter handling
- Reproducibility with fixed random seeds

## Important Disclaimers

- **Simulated streaming**: This is a simulated stream, not a production real-time system.
- **Small dataset**: The Breast Cancer Wisconsin dataset is modest in size, limiting generalization.
- **CPU-only quantum simulation**: Running quantum circuits on a classical simulator does NOT establish quantum speedup or advantage.
- **Different update strategies**: The classical model uses incremental updates (`partial_fit`), while the quantum model is trained on an initial window. These are not equivalent comparisons.

## Known Limitations

See `docs/LIMITATIONS.md` for detailed discussion of:
- Quantum simulation overhead
- Classical feature reduction
- Batch size and stream-length constraints
- Timing methodology dependencies on hardware and software

## Troubleshooting

### PennyLane import fails
Ensure you have installed the correct version from `requirements.txt`. If compatibility issues arise, consult the [PennyLane documentation](https://pennylane.ai/).

### Tests timeout
Reduce the number of optimization iterations or circuit depth in `configs/experiment.yaml`.

### Memory issues
Reduce batch size, stream length, or quantum circuit depth in the experiment configuration.

## Citation

If you use this project in academic work, please cite:
```
Quantum vs Classical Approaches for Streaming Data Analytics: An Experimental Study
[Your Institution], [Year]
```

## License

MIT

---

**Project Status**: Foundation phase complete. Ready for implementation.
