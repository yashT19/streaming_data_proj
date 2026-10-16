#!/usr/bin/env python
"""Verify full experiment results integrity."""
import pandas as pd
import numpy as np
import json

# Load full experiment results
raw = pd.read_csv('results/raw_results.csv')

print("="*80)
print("FULL EXPERIMENT RESULTS INSPECTION")
print("="*80)

print("\n1. RAW RESULTS SHAPE AND STATUS")
print(f"   Rows: {len(raw)}, Columns: {len(raw.columns)}")
print(f"   Status distribution: {raw['status'].value_counts().to_dict()}")

print("\n2. FAILED RUNS")
failed = raw[raw['status'] != 'success']
if len(failed) > 0:
    print(f"   FOUND {len(failed)} FAILURES:")
    for idx, row in failed.iterrows():
        print(f"   - {row['model_name']}: {row['error_message']}")
else:
    print("   ✓ No failures")

print("\n3. EXPERIMENT BREAKDOWN")
for exp_id in sorted(raw['experiment_id'].unique()):
    subset = raw[raw['experiment_id'] == exp_id]
    print(f"\n   {exp_id}:")
    print(f"     Total runs: {len(subset)}")
    print(f"     Models: {subset['model_name'].unique().tolist()}")
    print(f"     Batch sizes: {sorted(subset['batch_size'].unique())}")
    print(f"     Stream lengths: {sorted(subset['stream_length'].unique())}")
    print(f"     Repetitions per config: {subset.groupby(['model_name', 'batch_size', 'stream_length']).size().unique()}")

print("\n4. METRIC VALIDITY CHECK")
metric_cols = [c for c in raw.columns if c.startswith('metric_')]
print(f"   Metric columns: {len(metric_cols)}")
for col in metric_cols:
    values = raw[col].dropna().values
    if len(values) > 0:
        has_nan = np.isnan(values).any()
        has_inf = np.isinf(values).any()
        in_range = np.all((values >= 0) & (values <= 1))
        status = "✓" if (not has_nan and not has_inf and in_range) else "✗"
        print(f"   {status} {col}: min={values.min():.4f}, max={values.max():.4f}, count={len(values)}")

print("\n5. TIMING VALIDITY CHECK")
timing_cols = ['model_training_time_seconds', 'batch_update_time_seconds', 'prediction_time_seconds', 'end_to_end_time_seconds']
for col in timing_cols:
    values = raw[col].dropna().values
    if len(values) > 0:
        all_positive = np.all(values >= 0)
        all_finite = np.all(np.isfinite(values))
        status = "✓" if (all_positive and all_finite) else "✗"
        print(f"   {status} {col}: min={values.min():.6f}s, max={values.max():.6f}s, count={len(values)}")

print("\n6. TEST SET CONSISTENCY")
test_records = raw['test_records'].unique()
print(f"   Expected test records: 114")
print(f"   Actual: {test_records}")
print(f"   ✓ Consistent" if len(test_records) == 1 and test_records[0] == 114 else "   ✗ INCONSISTENT")

print("\n7. REPRODUCIBILITY CHECK (Experiment A - should be identical)")
exp_a = raw[raw['experiment_id'] == 'A_performance']
for model in exp_a['model_name'].unique():
    subset = exp_a[exp_a['model_name'] == model]
    acc_vals = subset['metric_accuracy'].values
    f1_vals = subset['metric_f1'].values
    print(f"   {model}: accuracy={acc_vals}, F1={f1_vals}")
    if len(acc_vals) > 1:
        acc_std = np.std(acc_vals)
        f1_std = np.std(f1_vals)
        status = "✓ Deterministic" if (acc_std < 1e-6 and f1_std < 1e-6) else f"⚠ Variation: acc_std={acc_std:.6f}, f1_std={f1_std:.6f}"
        print(f"     {status}")

print("\n8. BATCH SENSITIVITY (Experiment B)")
exp_b = raw[raw['experiment_id'] == 'B_batch_sensitivity']
if len(exp_b) > 0:
    print(f"   Classical batch sensitivity:")
    for batch_size in sorted(exp_b['batch_size'].unique()):
        subset = exp_b[(exp_b['batch_size'] == batch_size) & (exp_b['model_name'] == 'classical')]
        if len(subset) > 0:
            acc_mean = subset['metric_accuracy'].mean()
            time_mean = subset['end_to_end_time_seconds'].mean()
            print(f"     Batch {batch_size}: accuracy={acc_mean:.4f}, e2e_time={time_mean:.4f}s")
    print(f"   Quantum batch sensitivity:")
    for batch_size in sorted(exp_b['batch_size'].unique()):
        subset = exp_b[(exp_b['batch_size'] == batch_size) & (exp_b['model_name'] == 'quantum')]
        if len(subset) > 0:
            acc_mean = subset['metric_accuracy'].mean()
            time_mean = subset['end_to_end_time_seconds'].mean()
            print(f"     Batch {batch_size}: accuracy={acc_mean:.4f}, e2e_time={time_mean:.4f}s")

print("\n9. STREAM SENSITIVITY (Experiment C)")
exp_c = raw[raw['experiment_id'] == 'C_stream_sensitivity']
if len(exp_c) > 0:
    print(f"   Classical stream sensitivity:")
    for stream_len in sorted(exp_c['stream_length'].unique()):
        subset = exp_c[(exp_c['stream_length'] == stream_len) & (exp_c['model_name'] == 'classical')]
        if len(subset) > 0:
            acc_mean = subset['metric_accuracy'].mean()
            print(f"     Stream {stream_len}: accuracy={acc_mean:.4f}")
    print(f"   Quantum stream sensitivity:")
    for stream_len in sorted(exp_c['stream_length'].unique()):
        subset = exp_c[(exp_c['stream_length'] == stream_len) & (exp_c['model_name'] == 'quantum')]
        if len(subset) > 0:
            acc_mean = subset['metric_accuracy'].mean()
            print(f"     Stream {stream_len}: accuracy={acc_mean:.4f}")

print("\n10. SUMMARY STATISTICS BY EXPERIMENT AND MODEL")
summary = raw[raw['status'] == 'success'].groupby(['experiment_id', 'model_name']).agg({
    'metric_accuracy': ['mean', 'std'],
    'metric_f1': ['mean', 'std'],
    'end_to_end_time_seconds': ['mean', 'std'],
    'batch_update_time_seconds': ['mean', 'std'],
}).round(4)
print(summary)

print("\n" + "="*80)
print("FULL EXPERIMENT VERIFICATION: PASS ✓" if (raw['status'] == 'success').all() else "FULL EXPERIMENT VERIFICATION: FAIL ✗")
print(f"Total successful runs: {(raw['status'] == 'success').sum()} / {len(raw)}")
print("="*80)
