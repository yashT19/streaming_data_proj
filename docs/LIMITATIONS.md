# Known Limitations

This document outlines fundamental and practical limitations of this study that must be considered when interpreting results.

## Fundamental Constraints

### 1. Simulated vs. Real-Time Streaming
- This project simulates streaming via batch-oriented processing
- Not a production real-time system with true latency constraints
- Batch completion time is measured, not true streaming latency

### 2. Small Dataset Size
- Breast Cancer Wisconsin: 569 samples total
- After train/val/test split: ~340 training, ~114 validation, ~114 test
- Limited dataset reduces generalizability to larger or more complex problems

### 3. Reduced Quantum Feature Dimensionality
- Quantum model: 4 features (via PCA)
- Classical baseline (primary): 4 features (same for fair comparison)
- Classical model (secondary): All 30 features (shows classical potential)
- Quantum advantage often requires specific high-dimensional structures

### 4. CPU-Only Quantum Simulation
- PennyLane default.qubit simulator runs on CPU
- No quantum speedup or advantage is possible in this configuration
- Demonstrates algorithmic correctness, not quantum computational advantage
- All timing comparisons are between two classical algorithms

### 5. Different Model Update Strategies
- **Classical model**: Continuous incremental updates via `partial_fit()`
- **Quantum model**: Trained on initial window; later used for inference only
- Not equivalent comparison; classical model "learns" from entire stream
- A separate periodic-retraining experiment for quantum would be needed for fairness

### 6. Training Data Limitations
- Stream lengths (100, 200, 400 records) are small
- Limited opportunity for concept drift to manifest
- Insufficient samples to demonstrate online learning advantages

---

## Practical Limitations

### 1. Hyperparameter Tuning
- Models were not extensively tuned for this specific dataset
- Both models use reasonable defaults from literature
- Different tuning could substantially change relative performance
- This is intentional: report honest performance, not optimized results

### 2. Optimization Convergence
- Quantum optimizer may not fully converge on small datasets
- Classical model may reach a local minimum
- No guarantee of global optimality for either approach

### 3. Timing Methodology Dependence
- All timings depend on:
  - Hardware (CPU model, clock speed)
  - Software (Python version, NumPy/SciPy compilation flags)
  - Simulator state (warm vs. cold startup)
  - System load and competing processes
- Results are specific to this environment and not directly transferable

### 4. Reproducibility Constraints
- Fixed random seeds ensure reproducibility on same machine
- Different platforms or Python versions may produce slightly different numeric results
- Stochastic optimization may introduce small variance even with fixed seeds

### 5. Memory and Computational Budget
- Quantum circuit depth and optimization iterations are modest
- Larger circuits or more iterations could exceed available memory
- Trade-off between experimental scope and computational feasibility

### 6. Test Set Integrity
- Test set is held fixed throughout all experiments
- No hyperparameter tuning against test set
- Results are not optimistically biased

---

## Why This Study Does NOT Establish Quantum Advantage

### 1. Simulation Overhead
- Simulating quantum circuits classically requires exponential memory in worst case
- Even with efficient simulators, overhead is substantial
- Any timing advantage of quantum is fully negated by simulation

### 2. Problem Structure
- Tabular classification on small datasets does not inherently favor quantum approaches
- Quantum advantage requires specific properties:
  - High-dimensional feature spaces
  - Non-linear structures exploitable by entanglement
  - Problem sizes where quantum speedup theorems apply

### 3. Hybrid Approach
- Quantum classifiers still require classical optimization
- Training cost includes classical optimizer + quantum circuit evaluation
- No pure quantum advantage emerges

### 4. Sample Complexity
- Quantum models require similar or more training samples for same accuracy
- No advantage in sample efficiency observed in this regime

### 5. Generalization
- Results on Breast Cancer dataset not generalizable to:
  - Different problem types (e.g., regression, clustering)
  - Different scales (more samples, more features)
  - Real quantum hardware (has different noise characteristics)

---

## How to Interpret Results

### What We Can Conclude
- Relative performance on this specific dataset and configuration
- Implementation correctness (model produces valid predictions)
- Computational cost of simulation relative to classical methods
- Streaming behavior with fixed batch sizes

### What We CANNOT Conclude
- Quantum approaches are inherently better or worse for streaming data
- Quantum speedup or advantage (impossible with classical simulator)
- Generalizability to production systems or other domains
- Performance on quantum hardware (simulator is fundamentally different)

---

## Suggestions for Future Work

To overcome these limitations:

1. **Larger Datasets**: Use higher-dimensional streaming datasets
2. **Real Quantum Hardware**: Compare against actual quantum devices
3. **Problem Diversity**: Test on multiple problem types
4. **Theoretical Analysis**: Include complexity analysis alongside empirical results
5. **Concept Drift**: Implement synthetic drift and measure online adaptation
6. **Alternative Models**: Compare against other classical streaming methods (Bagging, Hoeffding Trees)
7. **Cost Analysis**: Include complete end-to-end cost (data ingestion, preprocessing, prediction, storage)

---

**Last Updated**: 2026-10-16
