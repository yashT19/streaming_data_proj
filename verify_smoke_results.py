#!/usr/bin/env python
"""Verify smoke test results integrity."""
import pandas as pd
import numpy as np
import json

# Load smoke test results
raw = pd.read_csv('results/smoke_raw_results.csv')

print("="*80)
print("SMOKE TEST RESULTS INSPECTION")
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

print("\n3. METRIC VALIDITY CHECK")
metric_cols = [c for c in raw.columns if c.startswith('metric_')]
print(f"   Metric columns: {len(metric_cols)}")
for col in metric_cols:
    values = raw[col].values
    has_nan = np.isnan(values).any()
    has_inf = np.isinf(values).any()
    in_range = np.all((values >= 0) & (values <= 1))
    status = "✓" if (not has_nan and not has_inf and in_range) else "✗"
    print(f"   {status} {col}: min={values.min():.4f}, max={values.max():.4f}, has_nan={has_nan}, has_inf={has_inf}")

print("\n4. TIMING VALIDITY CHECK")
timing_cols = ['model_training_time_seconds', 'batch_update_time_seconds', 'prediction_time_seconds', 'end_to_end_time_seconds']
for col in timing_cols:
    values = raw[col].values
    all_positive = np.all(values >= 0)
    all_finite = np.all(np.isfinite(values))
    status = "✓" if (all_positive and all_finite) else "✗"
    print(f"   {status} {col}: min={values.min():.6f}s, max={values.max():.6f}s")

print("\n5. TEST SET CONSISTENCY")
test_records = raw['test_records'].unique()
print(f"   Expected test records: 114")
print(f"   Actual: {test_records}")
print(f"   ✓ Consistent" if len(test_records) == 1 and test_records[0] == 114 else "   ✗ INCONSISTENT")

print("\n6. DETAILED RESULTS BY MODEL")
for model in raw['model_name'].unique():
    subset = raw[raw['model_name'] == model]
    print(f"\n   {model.upper()}:")
    print(f"     Accuracy:  {subset['metric_accuracy'].values[0]:.4f}")
    print(f"     Precision: {subset['metric_precision'].values[0]:.4f}")
    print(f"     Recall:    {subset['metric_recall'].values[0]:.4f}")
    print(f"     F1:        {subset['metric_f1'].values[0]:.4f}")
    print(f"     Train time: {subset['model_training_time_seconds'].values[0]:.6f}s")
    print(f"     End-to-end: {subset['end_to_end_time_seconds'].values[0]:.6f}s")

print("\n7. EXPERIMENT METADATA")
with open('results/experiment_metadata.json', 'r') as f:
    meta = json.load(f)
    for item in meta:
        print(f"   Mode: {item['mode']}")
        print(f"   Test records: {item['test_records']}")
        print(f"   Feature count: {item['feature_count']}")
        print(f"   Quantum iterations: {item['quantum_iterations']}")
        print(f"   Timing repetitions: {item['timing_repetitions']}")

print("\n8. COLUMN STRUCTURE")
print(f"   Total columns: {len(raw.columns)}")
print("   Key columns present:")
required = ['experiment_id', 'model_name', 'status', 'metric_accuracy', 'metric_f1',
            'model_training_time_seconds', 'prediction_time_seconds', 'test_records']
for col in required:
    status = "✓" if col in raw.columns else "✗"
    print(f"   {status} {col}")

print("\n" + "="*80)
print("SMOKE TEST VERIFICATION: PASS ✓" if (raw['status'] == 'success').all() else "SMOKE TEST VERIFICATION: FAIL ✗")
print("="*80)
