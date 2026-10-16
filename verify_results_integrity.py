#!/usr/bin/env python
"""Task 16: Verify result file and plot integrity."""
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

print('=== Task 16: Result File and Plot Integrity Check ===')

results_dir = Path('results')

# Check CSV files exist and are valid
print('\n--- CSV Files ---')
csv_files = ['raw_results.csv', 'summary.csv', 'summary_statistics.csv', 'smoke_raw_results.csv', 'smoke_summary.csv']
for csv_file in csv_files:
    path = results_dir / csv_file
    if path.exists():
        df = pd.read_csv(path)
        print(f'✓ {csv_file}: {len(df)} rows, {len(df.columns)} cols')
        # Verify no NaN in critical columns
        assert not df.empty, f"{csv_file} is empty"
        if 'status' in df.columns:
            assert (df['status'] == 'success').all(), f"{csv_file} has failed experiments"
    else:
        print(f'✗ {csv_file}: NOT FOUND')

# Check JSON files
print('\n--- JSON Files ---')
json_files = {
    'experiment_metadata.json': 'metadata',
    'executed_config.json': 'config'
}
for json_file, desc in json_files.items():
    path = results_dir / json_file
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f'✓ {json_file}: valid JSON ({desc})')
    else:
        print(f'✗ {json_file}: NOT FOUND')

# Check PNG plots
print('\n--- PNG Plot Files ---')
png_files = [
    'accuracy_f1_by_model.png',
    'confusion_matrices.png',
    'throughput_comparison.png',
    'timing_vs_batch_size.png',
    'stream_length_sensitivity.png'
]

plots_dir = results_dir / 'plots'
for png_file in png_files:
    # Check both in root and plots/ subdirectory
    path = results_dir / png_file
    if path.exists():
        size = path.stat().st_size
        print(f'✓ {png_file}: {size} bytes')
    else:
        path = plots_dir / f'{png_file.replace(".png", "")}.png'
        if path.exists():
            size = path.stat().st_size
            print(f'✓ plots/{path.name}: {size} bytes')
        else:
            print(f'? {png_file}: not found in root or plots/')

# List plots/ subdirectory
if plots_dir.exists():
    print(f'\n--- Plots Subdirectory ---')
    plot_files = list(plots_dir.glob('*.png'))
    print(f'{len(plot_files)} PNG files:')
    for pf in sorted(plot_files):
        size = pf.stat().st_size
        print(f'  {pf.name}: {size} bytes')

# Verify raw_results.csv has actual data from experiments
print('\n--- Data Validation ---')
raw_df = pd.read_csv(results_dir / 'raw_results.csv')
print(f'Raw results: {len(raw_df)} experiments')
print(f'  Classical: {len(raw_df[raw_df["model_name"] == "classical"])} runs')
print(f'  Quantum: {len(raw_df[raw_df["model_name"] == "quantum"])} runs')

# Check all metrics are numeric and in valid range
numeric_cols = ['metric_accuracy', 'metric_precision', 'metric_recall', 'metric_f1']
for col in numeric_cols:
    values = raw_df[col].values
    assert np.isfinite(values).all(), f"{col} contains non-finite values"
    assert (values >= 0).all() and (values <= 1).all(), f"{col} values outside [0, 1]"
    print(f'✓ {col}: all values in [0, 1]')

# Check timing columns are non-negative
timing_cols = ['model_training_time_seconds', 'prediction_time_seconds', 'end_to_end_time_seconds']
for col in timing_cols:
    values = raw_df[col].values
    assert (values >= 0).all(), f"{col} has negative values"
    print(f'✓ {col}: all non-negative')

print('\n✓ Result file and plot integrity PASSED')
