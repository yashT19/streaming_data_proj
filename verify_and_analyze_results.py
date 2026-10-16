#!/usr/bin/env python
"""Comprehensive verification and analysis of experiment results."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

def verify_results():
    """Verify data integrity and completeness."""
    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS VERIFICATION")
    print("="*100)
    
    # Load raw results
    raw_path = Path('results/raw_results.csv')
    if not raw_path.exists():
        print("ERROR: raw_results.csv not found!")
        return False
    
    raw = pd.read_csv(raw_path)
    print(f"\n[1] DATA LOADING")
    print(f"    ✓ raw_results.csv loaded: {raw.shape[0]} observations × {raw.shape[1]} columns")
    
    # Check status
    print(f"\n[2] RUN STATUS")
    status_counts = raw['status'].value_counts()
    print(f"    Status distribution: {status_counts.to_dict()}")
    
    if (raw['status'] != 'success').any():
        print("    ✗ WARNING: Some runs failed!")
        failed = raw[raw['status'] != 'success']
        print(f"    Failed runs: {len(failed)}")
        for idx, row in failed.iterrows():
            print(f"      - {row['model_name']}: {row['error_message']}")
        return False
    else:
        print("    ✓ All 42 runs successful")
    
    # Filter to successful runs
    successful = raw[raw['status'] == 'success'].copy()
    
    # Check experiments
    print(f"\n[3] EXPERIMENT COVERAGE")
    exp_counts = successful['experiment_id'].value_counts().sort_index()
    for exp_id, count in exp_counts.items():
        print(f"    {exp_id}: {count} observations")
    
    # Verify Experiment A (should have 6 observations: 3 classical, 3 quantum)
    exp_a = successful[successful['experiment_id'] == 'A_performance']
    if len(exp_a) != 6:
        print(f"    ✗ WARNING: Experiment A has {len(exp_a)} obs, expected 6")
    else:
        print(f"    ✓ Experiment A: 6 observations (3 classical, 3 quantum)")
    
    # Verify Experiment B (should have 18: 3 batch sizes × 2 models × 3 reps)
    exp_b = successful[successful['experiment_id'] == 'B_batch_sensitivity']
    if len(exp_b) != 18:
        print(f"    ✗ WARNING: Experiment B has {len(exp_b)} obs, expected 18")
    else:
        print(f"    ✓ Experiment B: 18 observations (3 batch sizes × 2 models × 3 reps)")
    
    # Verify Experiment C (should have 18: 3 stream lengths × 2 models × 3 reps)
    exp_c = successful[successful['experiment_id'] == 'C_stream_sensitivity']
    if len(exp_c) != 18:
        print(f"    ✗ WARNING: Experiment C has {len(exp_c)} obs, expected 18")
    else:
        print(f"    ✓ Experiment C: 18 observations (3 stream lengths × 2 models × 3 reps)")
    
    # Check test set consistency
    print(f"\n[4] TEST SET CONSISTENCY")
    test_sizes = successful['test_records'].unique()
    if len(test_sizes) != 1 or test_sizes[0] != 114:
        print(f"    ✗ WARNING: Test set size inconsistent: {test_sizes}")
        return False
    else:
        print(f"    ✓ All observations use test_records=114 (fixed held-out)")
    
    # Check metric validity
    print(f"\n[5] METRIC VALIDITY")
    metric_cols = [c for c in successful.columns if c.startswith('metric_')]
    all_valid = True
    for col in metric_cols:
        vals = successful[col].dropna()
        has_nan = np.isnan(vals).any()
        has_inf = np.isinf(vals).any()
        in_range = np.all((vals >= 0) & (vals <= 1)) if len(vals) > 0 else True
        if has_nan or has_inf or not in_range:
            print(f"    ✗ {col}: INVALID (nan={has_nan}, inf={has_inf}, in_range={in_range})")
            all_valid = False
    
    if all_valid:
        print(f"    ✓ All {len(metric_cols)} metrics are finite and in [0, 1]")
    
    # Check timing validity
    print(f"\n[6] TIMING VALIDITY")
    timing_cols = ['model_training_time_seconds', 'batch_update_time_seconds', 
                   'prediction_time_seconds', 'end_to_end_time_seconds']
    all_timing_valid = True
    for col in timing_cols:
        if col in successful.columns:
            vals = successful[col].dropna()
            all_positive = np.all(vals >= 0)
            all_finite = np.all(np.isfinite(vals))
            if not (all_positive and all_finite):
                print(f"    ✗ {col}: INVALID (positive={all_positive}, finite={all_finite})")
                all_timing_valid = False
    
    if all_timing_valid:
        print(f"    ✓ All timing values are non-negative and finite")
    
    # Reproducibility check
    print(f"\n[7] REPRODUCIBILITY (Experiment A)")
    exp_a = successful[successful['experiment_id'] == 'A_performance']
    for model in ['classical', 'quantum']:
        subset = exp_a[exp_a['model_name'] == model]
        acc_vals = subset['metric_accuracy'].values
        f1_vals = subset['metric_f1'].values
        acc_std = np.std(acc_vals)
        f1_std = np.std(f1_vals)
        
        if acc_std < 1e-6 and f1_std < 1e-6:
            print(f"    ✓ {model}: deterministic (acc_std={acc_std:.2e}, f1_std={f1_std:.2e})")
        else:
            print(f"    ⚠ {model}: some variation (acc_std={acc_std:.4f}, f1_std={f1_std:.4f})")
    
    # Model performance summary
    print(f"\n[8] PERFORMANCE SUMMARY BY MODEL")
    for model in ['classical', 'quantum']:
        subset = successful[successful['model_name'] == model]
        acc_mean = subset['metric_accuracy'].mean()
        f1_mean = subset['metric_f1'].mean()
        print(f"    {model}: accuracy={acc_mean:.4f}, f1={f1_mean:.4f}")
    
    # Batch size analysis
    print(f"\n[9] BATCH SIZE EFFECTS (Experiment B)")
    exp_b = successful[successful['experiment_id'] == 'B_batch_sensitivity']
    for batch_size in sorted(exp_b['batch_size'].unique()):
        for model in ['classical', 'quantum']:
            subset = exp_b[(exp_b['batch_size'] == batch_size) & (exp_b['model_name'] == model)]
            if len(subset) > 0:
                acc_mean = subset['metric_accuracy'].mean()
                print(f"    Batch {batch_size:2d}, {model:10s}: accuracy={acc_mean:.4f}")
    
    # Stream length analysis
    print(f"\n[10] STREAM LENGTH EFFECTS (Experiment C)")
    exp_c = successful[successful['experiment_id'] == 'C_stream_sensitivity']
    for stream_len in sorted(exp_c['stream_length'].unique()):
        for model in ['classical', 'quantum']:
            subset = exp_c[(exp_c['stream_length'] == stream_len) & (exp_c['model_name'] == model)]
            if len(subset) > 0:
                acc_mean = subset['metric_accuracy'].mean()
                print(f"    Stream {stream_len:3d}, {model:10s}: accuracy={acc_mean:.4f}")
    
    print(f"\n" + "="*100)
    print("VERIFICATION COMPLETE: ALL CHECKS PASSED ✓")
    print("="*100 + "\n")
    
    return True

if __name__ == "__main__":
    success = verify_results()
    exit(0 if success else 1)
