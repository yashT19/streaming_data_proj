# Project: Quantum vs Classical Approaches for Streaming Data Analytics — An Experimental Study

## 1. Your role

Act as a senior Python engineer, quantum machine learning researcher, experimental scientist, and software test engineer.

Your task is to build a complete, reproducible, well-tested academic project comparing classical and quantum machine learning approaches for simulated streaming data classification.

Work autonomously where reasonable. Inspect the repository, create a plan, implement the project in small stages, run actual commands and tests, inspect their outputs, fix failures, and verify the final results.

Do not merely generate code and claim that it works. Execute the code wherever the environment permits and report the evidence for successful execution.

Prioritize a small, correct, reproducible project over unnecessary complexity. Do not overengineer it.

## 2. Project objective

Project title:

"Quantum vs Classical Approaches for Streaming Data Analytics: An Experimental Study"

Investigate the practical trade-offs between a classical incremental machine learning model and a small quantum machine learning model when processing data in sequential batches.

The project must answer:

1. How do predictive performance metrics compare?
2. How do training, updating, and inference costs compare?
3. How do batch size and stream length affect performance?
4. What limitations prevent us from concluding that a quantum approach is universally better?
5. Optionally, how do the models behave under simulated concept drift?

The objective is an honest experimental comparison, not proving quantum superiority.

A classical quantum-circuit simulator running on a CPU is not evidence of quantum computational speedup. State this explicitly in the documentation.

## 3. Constraints and scope

Keep the project manageable for a student team.

Required:
- Python.
- NumPy and Pandas where useful.
- scikit-learn.
- One quantum framework: prefer PennyLane or Qiskit, choosing whichever provides the simplest reliable implementation in the current environment.
- A quantum simulator; no quantum hardware is required.
- Matplotlib for plots.
- pytest for automated testing.
- CSV or JSON files for machine-readable experimental results.

Use the Breast Cancer Wisconsin Diagnostic dataset through scikit-learn as the initial dataset.

Do not introduce Kafka, Spark, cloud infrastructure, databases, a web interface, distributed processing, or other unnecessary components.

Do not add dependencies without a clear reason. Pin or constrain dependencies appropriately and document the tested Python version.

Keep the quantum model small, initially targeting four input features and four qubits.

If the chosen quantum framework is unavailable or incompatible, diagnose the problem, try a reasonable compatible version or alternative simulator, and document the resolution. Never silently replace the quantum model with a classical model.

## 4. Phase 0 — Inspect the environment before modifying anything

Before implementation:

1. Inspect the current working directory, existing source files, Git status, Python version, and installed dependencies.
2. Preserve existing user work. Do not overwrite or delete files without inspecting them first.
3. Determine whether a virtual environment already exists and reuse it if appropriate.
4. Check the available memory and whether a CPU-only implementation is necessary.
5. Check the installed quantum framework and its simulator APIs.
6. Identify any dependency conflicts before installing packages.

Do not perform destructive Git operations, expose credentials, or modify unrelated projects.

If the repository already contains a partial implementation, audit and reuse working components rather than rebuilding everything blindly.

Create `docs/IMPLEMENTATION_PLAN.md` containing:
- Architecture.
- Implementation stages.
- Expected outputs.
- Test strategy.
- Major risks.
- Estimated computational cost.
- Known limitations.

Then proceed with implementation without waiting for approval unless a genuinely blocking decision requires user input.

## 5. Phase 1 — Design a clean project structure

Use a structure similar to:

quantum-streaming-study/
    README.md
    requirements.txt
    pyproject.toml
    .gitignore
    configs/
        experiment.yaml
    src/
        __init__.py
        data_processing.py
        stream_simulator.py
        classical_model.py
        quantum_model.py
        evaluation.py
        experiment_runner.py
        plotting.py
        reproducibility.py
    tests/
        test_data_processing.py
        test_stream_simulator.py
        test_classical_model.py
        test_quantum_model.py
        test_evaluation.py
        test_experiment_runner.py
    notebooks/
        exploration.ipynb
    results/
        .gitkeep
    docs/
        IMPLEMENTATION_PLAN.md
        METHODOLOGY.md
        RESULTS.md
        LIMITATIONS.md

Adapt this structure if a simpler design is clearly better.

Requirements:
- Keep core logic in importable Python modules.
- Keep notebooks optional, not essential for executing the experiments.
- Separate data preparation, models, evaluation, orchestration, and plotting.
- Provide a single documented command to run the main experiment.
- Avoid duplicated code and hidden global state.
- Use type hints and clear docstrings for important functions.
- Make configuration explicit rather than scattering constants throughout the code.

## 6. Phase 2 — Data preparation and leakage prevention

Load the Breast Cancer Wisconsin Diagnostic dataset using scikit-learn.

Implement:

1. Dataset loading and validation.
2. Feature and target separation.
3. Stratified train, validation, and test splitting.
4. Reproducible shuffling and fixed random seeds.
5. Classical feature preprocessing.
6. Quantum feature reduction to four input features.
7. Feature scaling suitable for the selected quantum encoding.

Use a fixed, documented split, such as 60% training, 20% validation, and 20% testing, with stratification.

Fit all preprocessing transformations using training data only. Never fit a scaler, feature selector, dimensionality reduction method, or other learned transformation on the test data.

If PCA or another learned feature reducer is used, fit it on training data only and apply the fitted transformation to validation and test data.

For the primary fair comparison, evaluate both models using the same reduced four-feature representation and the same held-out test records. A separate experiment using all available features for the classical model may be included, but must be labeled as a different comparison.

Add checks for:
- Empty datasets.
- Invalid dimensions.
- Missing or non-finite values.
- Incorrect feature-target lengths.
- Incorrect class labels.
- Overlapping train, validation, and test record indices.
- Deterministic preprocessing with fixed seeds.
- Correct transformation of validation and test records.

Document any assumptions about the data.

## 7. Phase 3 — Implement a deterministic streaming simulator

Create a reusable simulator that yields sequential batches from the training data.

Requirements:
- Support configurable batch sizes, initially 16, 32, and 64 where valid.
- Preserve the selected record order after the initial reproducible shuffle.
- Include a smaller final batch if the number of records is not divisible by the batch size.
- Never silently discard records.
- Never include test records in training batches.
- Preserve feature-target alignment.
- Report batch index, number of records, and cumulative records processed.
- Validate invalid batch sizes.
- Support reproducible experiments.
- Track the number of records emitted and verify it matches the intended stream length.

Test edge cases such as one batch, an incomplete final batch, and a batch size larger than the stream.

Important distinction: this is a simulated stream, not a production real-time streaming system. State this in the report.

## 8. Phase 4 — Implement the classical baseline

Implement a scikit-learn `SGDClassifier` or another justified incremental classifier.

Prefer `SGDClassifier` with a documented loss function and fixed random seed.

Requirements:
- Use `partial_fit()` for incremental updates.
- Supply the full set of classes on the initial update.
- Handle the first batch correctly.
- Support configurable batch sizes.
- Measure update time separately from prediction time.
- Evaluate on the fixed held-out test set without using it for updates.
- Record cumulative training time and per-batch metrics where applicable.
- Support reproducible runs.
- Return structured results rather than printing metrics only.

Test that:
- The model can train on the first batch.
- Subsequent batches update it successfully.
- Predictions have the correct length and valid labels.
- Repeated runs with the same configuration are reproducible to a reasonable tolerance.
- Invalid input dimensions fail with useful error messages.

## 9. Phase 5 — Implement the quantum classifier

Implement a real quantum circuit classifier using the selected framework and simulator.

Use a small variational quantum circuit with:
- Four qubits initially.
- Four normalized input features.
- Angle encoding.
- Parameterized trainable gates.
- Entangling gates.
- Measurement expectation values or an equivalent documented output.
- A classical optimizer for training the circuit parameters.

Requirements:
1. Document the circuit architecture.
2. Explain how inputs are encoded.
3. Explain how measured outputs are converted into predictions.
4. Use a fixed random seed wherever the framework permits it.
5. Expose circuit depth, optimization settings, and shot configuration when relevant.
6. Use a deterministic simulator configuration for the main baseline when possible.
7. Record the framework, simulator, execution mode, and important settings.
8. Measure training and inference time separately.
9. Validate feature dimensions and numerical ranges.
10. Keep the initial training configuration computationally modest.
11. Avoid unnecessary circuit depth and excessive optimization iterations.
12. Add timeouts or iteration limits where feasible to avoid unbounded runs.

The quantum model must execute an actual quantum circuit through the selected simulator. Do not create a fake quantum model that simply calls a classical classifier.

The quantum circuit classifier does not need to support `partial_fit()`. For the minimum viable study, train it on an initial training window and evaluate it as later batches arrive.

Clearly document that this differs from the continuously updated classical model. Do not present the two update strategies as equivalent.

If time permits, add a separate periodic-retraining experiment for the quantum model. Report its retraining cost independently.

Add tests for:
- Circuit construction.
- Correct number of qubits.
- Valid parameter shapes.
- Finite circuit outputs.
- Successful optimization on a tiny synthetic dataset.
- Valid prediction labels and output shapes.
- Reproducibility within reasonable numerical tolerance.
- Useful failure messages for invalid input.

The quantum model may perform worse than the classical baseline. Do not alter the test set or tune repeatedly against it to force a desired result.

## 10. Phase 6 — Experimental design

Implement the following minimum experiments.

Experiment A: Predictive performance
- Train both approaches according to their documented strategies.
- Evaluate on the same fixed held-out test records.
- Calculate accuracy, precision, recall, F1-score, and confusion matrices.

Experiment B: Batch-size sensitivity
- Test batch sizes 16, 32, and 64 for the streaming pipeline, where valid.
- Measure classical update time and end-to-end processing time.
- For the quantum model, measure inference cost across corresponding batch sizes and report training cost separately.
- Do not imply that the quantum model performs incremental updates if it does not.

Experiment C: Stream-length sensitivity
- Evaluate the effects of increasing training-stream lengths using reproducible prefixes or predetermined training subsets.
- Keep the test set fixed.
- Ensure preprocessing is fitted only on the permitted training portion.
- Explain whether the model is retrained for each stream-length experiment.

Experiment D: Optional concept-drift experiment
- Create a controlled synthetic distribution change or another defensible drift scenario.
- Do not modify the original test labels or leak test data into training.
- Clearly distinguish synthetic drift results from results on the original dataset.
- Implement this only after all required experiments work.

Use at least three timing repetitions where practical. Report the mean and standard deviation. Do not report failed or incomplete runs as successful results.

For reproducibility, use the same predefined splits, feature representations, and evaluation records for paired comparisons.

If the experiment is too computationally expensive, reduce circuit depth, optimization iterations, or the number of repeated runs. Document every reduction. Do not fabricate results.

## 11. Phase 7 — Metrics and fair benchmarking

Calculate and save:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Model training time.
- Model update time where applicable.
- Prediction latency.
- End-to-end processing time.
- Throughput in records per second.
- Peak memory usage when practical.

Define every metric in the documentation.

For timing:
- Use an appropriate monotonic high-resolution timer.
- Separate initialization and compilation/warm-up costs from steady-state execution when relevant.
- Report cold-start and warm inference costs separately when feasible.
- State whether quantum execution includes circuit construction, simulator execution, shot sampling, and classical post-processing.
- State whether batch latency includes data preparation.
- Use the same timing methodology across repetitions.
- Avoid comparing a warmed-up classical prediction against a quantum measurement that includes unrelated one-time setup without disclosing it.

For throughput, define the numerator and timing interval explicitly.

For memory, use the same measurement approach and explain whether it measures process-level memory or a narrower component.

If a metric is unavailable, report it as unavailable with a reason rather than inventing a value.

Do not assume quantum simulation is faster or more memory-efficient.

## 12. Phase 8 — Results storage and visualization

Store raw experiment records in CSV or JSON under `results/`.

Every experiment record must include enough metadata to reproduce it, including:
- Experiment ID.
- Model name and version.
- Dataset and split seed.
- Batch size and stream length.
- Number of input features and qubits.
- Circuit and optimizer configuration where relevant.
- Timing repetition number.
- All measured metrics.
- Framework versions.
- Execution environment information.
- Run status and any error message.

Generate at least these plots:
1. Accuracy and F1-score by model.
2. Processing or inference time against batch size.
3. Throughput comparison.
4. Optional concept-drift performance over time.

Requirements:
- Use clear titles, axis labels, units, and legends.
- Do not hide poor results.
- Do not use fabricated or manually entered metric values.
- Load plots from actual experiment result files.
- Verify that the expected files are created.
- Handle missing or failed experiment records explicitly.
- Keep raw measurements separate from aggregated summary tables.

Create `results/summary.csv` or an equivalent machine-readable summary.

## 13. Phase 9 — Automated quality assurance

Create a pytest suite covering the entire project.

At minimum:
- Data-loading tests.
- Data-leakage and split-integrity tests.
- Streaming-batch tests.
- Classical training and prediction tests.
- Quantum circuit and tiny-training tests.
- Metric calculation tests.
- Experiment output-schema tests.
- CSV/JSON serialization tests.
- Plot-generation smoke tests.
- Reproducibility tests.
- Invalid-input and edge-case tests.

Use small synthetic datasets for quick unit tests. Avoid making the full quantum optimization run every time the unit test suite executes.

Create a separate optional integration test that runs a small end-to-end quantum experiment.

Add a fast smoke-test command that checks that:
1. The dataset loads.
2. Preprocessing succeeds.
3. The stream emits all expected records.
4. The classical model trains and predicts.
5. The quantum circuit executes and produces valid predictions.
6. Metrics are finite and within valid ranges.
7. Result files are generated.
8. Plots can be created.

After implementation, run:
- The full unit test suite.
- The smoke test.
- A small end-to-end experiment.
- The main experiment if computationally practical.

Inspect all error messages and warnings. Fix errors introduced by your code and investigate relevant warnings. Do not suppress warnings merely to make the output look clean.

If a test cannot be executed because of missing dependencies, network restrictions, or resource limits, explicitly report that it was not run and why.

Never claim that a test passed unless you executed it and observed a successful result.

## 14. Phase 10 — Dependency and execution checks

Provide:
- A documented Python version.
- A dependency file with compatible versions or constraints.
- Clear virtual-environment setup instructions.
- A command to run the tests.
- A command to run a small smoke experiment.
- A command to run the full experiment.
- A command to generate plots from saved results.

Test these commands in a clean or isolated environment when feasible.

Avoid adding large optional dependencies unless required.

If installation fails:
1. Inspect the actual error.
2. Identify the incompatible dependency or environment assumption.
3. Make the smallest justified correction.
4. Retry the relevant installation or test.
5. Document unresolved issues.

Do not claim clean-environment compatibility unless you have verified it.

## 15. Phase 11 — Documentation and academic integrity

Produce the following documents.

README.md:
- Project overview.
- Research questions.
- Architecture.
- Prerequisites.
- Installation.
- Commands to run.
- Expected output files.
- Troubleshooting.
- Limitations.

docs/METHODOLOGY.md:
- Dataset and split.
- Preprocessing.
- Streaming simulation.
- Classical model.
- Quantum circuit.
- Training protocols.
- Evaluation metrics.
- Experimental configurations.
- Timing methodology.

docs/RESULTS.md:
- Actual measured results.
- Tables and interpretation.
- Figures or links to generated figures.
- Observed limitations.
- Differences between the models.

docs/LIMITATIONS.md:
- Simulated rather than real-time streaming.
- Small dataset.
- Reduced quantum input dimensionality.
- Classical simulation overhead.
- Different model-update strategies.
- Hardware and software dependence of timings.
- Limits on generalizing results.
- Why this experiment does not establish a quantum speedup.

Do not fabricate references, citations, paper titles, experimental results, hardware specifications, or benchmark numbers.

If you add academic references, verify that each reference exists and that its bibliographic information is accurate. Clearly distinguish published findings from your own results.

Do not write the report as if the experiment succeeded before running it. Use actual saved results and describe failures honestly.

## 16. Implementation workflow: mandatory checkpoints

Work through these checkpoints in order.

Checkpoint 1:
- Inspect the environment.
- Create the implementation plan.
- Define the architecture and commands.

Checkpoint 2:
- Implement dataset loading, preprocessing, and stream simulation.
- Run tests.
- Fix failures before proceeding.

Checkpoint 3:
- Implement the classical model.
- Run unit tests and a smoke experiment.
- Verify the baseline metrics.

Checkpoint 4:
- Implement the quantum circuit and classifier.
- Run circuit tests and a tiny optimization experiment.
- Verify actual circuit execution and prediction shapes.

Checkpoint 5:
- Integrate both models.
- Implement experimental orchestration and result serialization.
- Validate output schemas and leakage safeguards.

Checkpoint 6:
- Run the minimum experiments.
- Generate plots from real results.
- Inspect timing and metric anomalies.

Checkpoint 7:
- Run the full test suite and end-to-end smoke test.
- Fix failures.
- Complete documentation.
- Produce a final audit.

At each checkpoint:
1. Inspect the changed files.
2. Execute relevant tests.
3. Inspect the command output and exit status.
4. Fix discovered problems.
5. Re-run affected tests.
6. Record the result in `docs/IMPLEMENTATION_PLAN.md`.

Do not skip checkpoints merely because code has been generated.

If the environment cannot run a checkpoint, record it as BLOCKED, explain the reason, and continue only with tasks that do not depend on pretending it succeeded.

## 17. Final acceptance criteria

The project is complete only when the following can be verified:

[ ] The documented environment and dependencies are specified.
[ ] The dataset loads and preprocessing passes tests.
[ ] Train, validation, and test sets are separated correctly.
[ ] The simulated stream processes all intended training records.
[ ] The classical baseline trains and predicts successfully.
[ ] A real quantum circuit executes on the selected simulator.
[ ] The quantum classifier completes a small training run and predicts successfully.
[ ] Both models are evaluated on the same held-out records for the primary comparison.
[ ] The differences in training and update strategies are documented.
[ ] Metrics and timings are computed from actual runs.
[ ] Raw results and summary results are saved.
[ ] Required plots are generated from saved results.
[ ] Automated tests pass, or each unresolved failure is clearly documented.
[ ] The smoke experiment completes successfully, or is marked BLOCKED with evidence.
[ ] Setup and execution commands are documented and tested where feasible.
[ ] The report does not claim unverified quantum advantages.
[ ] No result, reference, or successful test is fabricated.

## 18. Final response to me

After finishing, provide a concise but evidence-based completion report containing:

1. What was implemented.
2. The final repository structure.
3. The exact installation and execution commands.
4. Tests actually run and their pass/fail counts.
5. Whether the quantum circuit successfully executed and trained.
6. Paths to the generated raw results, summary tables, and plots.
7. Actual experimental results, if the experiments ran.
8. Known issues and limitations.
9. Any checkpoint that remains blocked.
10. The next three most valuable actions, prioritizing essential fixes over optional enhancements.

Be honest about what was executed versus what was merely written.

Begin with Phase 0 now. Inspect the environment, make the implementation plan, and proceed checkpoint by checkpoint.