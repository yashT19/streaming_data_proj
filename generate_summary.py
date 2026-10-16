#!/usr/bin/env python
"""Generate aggregated summary statistics from raw results."""
import pandas as pd
import numpy as np

# Load raw results
raw = pd.read_csv('results/raw_results.csv')
successful = raw[raw['status'] == 'success'].copy()

# Create summary by experiment, model, and batch size/stream length
summary_rows = []

# Experiment A
exp_a = successful[successful['experiment_id'] == 'A_performance']
for model in ['classical', 'quantum']:
    subset = exp_a[exp_a['model_name'] == model]
    if len(subset) > 0:
        summary_rows.append({
            'experiment_id': 'A_performance',
            'model_name': model,
            'batch_size': 32,
            'stream_length': 341,
            'n_observations': len(subset),
            'accuracy_mean': subset['metric_accuracy'].mean(),
            'accuracy_std': subset['metric_accuracy'].std(),
            'precision_mean': subset['metric_precision'].mean(),
            'precision_std': subset['metric_precision'].std(),
            'recall_mean': subset['metric_recall'].mean(),
            'recall_std': subset['metric_recall'].std(),
            'f1_mean': subset['metric_f1'].mean(),
            'f1_std': subset['metric_f1'].std(),
            'training_time_mean': subset['model_training_time_seconds'].mean(),
            'training_time_std': subset['model_training_time_seconds'].std(),
            'update_time_mean': subset['batch_update_time_seconds'].mean(),
            'update_time_std': subset['batch_update_time_seconds'].std(),
            'prediction_time_mean': subset['prediction_time_seconds'].mean(),
            'prediction_time_std': subset['prediction_time_seconds'].std(),
            'e2e_time_mean': subset['end_to_end_time_seconds'].mean(),
            'e2e_time_std': subset['end_to_end_time_seconds'].std(),
            'throughput_mean': subset['throughput_records_per_second'].mean(),
            'throughput_std': subset['throughput_records_per_second'].std(),
        })

# Experiment B
exp_b = successful[successful['experiment_id'] == 'B_batch_sensitivity']
for batch_size in sorted(exp_b['batch_size'].unique()):
    for model in ['classical', 'quantum']:
        subset = exp_b[(exp_b['batch_size'] == batch_size) & (exp_b['model_name'] == model)]
        if len(subset) > 0:
            summary_rows.append({
                'experiment_id': 'B_batch_sensitivity',
                'model_name': model,
                'batch_size': batch_size,
                'stream_length': 341,
                'n_observations': len(subset),
                'accuracy_mean': subset['metric_accuracy'].mean(),
                'accuracy_std': subset['metric_accuracy'].std(),
                'precision_mean': subset['metric_precision'].mean(),
                'precision_std': subset['metric_precision'].std(),
                'recall_mean': subset['metric_recall'].mean(),
                'recall_std': subset['metric_recall'].std(),
                'f1_mean': subset['metric_f1'].mean(),
                'f1_std': subset['metric_f1'].std(),
                'training_time_mean': subset['model_training_time_seconds'].mean(),
                'training_time_std': subset['model_training_time_seconds'].std(),
                'update_time_mean': subset['batch_update_time_seconds'].mean(),
                'update_time_std': subset['batch_update_time_seconds'].std(),
                'prediction_time_mean': subset['prediction_time_seconds'].mean(),
                'prediction_time_std': subset['prediction_time_seconds'].std(),
                'e2e_time_mean': subset['end_to_end_time_seconds'].mean(),
                'e2e_time_std': subset['end_to_end_time_seconds'].std(),
                'throughput_mean': subset['throughput_records_per_second'].mean(),
                'throughput_std': subset['throughput_records_per_second'].std(),
            })

# Experiment C
exp_c = successful[successful['experiment_id'] == 'C_stream_sensitivity']
for stream_length in sorted(exp_c['stream_length'].unique()):
    for model in ['classical', 'quantum']:
        subset = exp_c[(exp_c['stream_length'] == stream_length) & (exp_c['model_name'] == model)]
        if len(subset) > 0:
            summary_rows.append({
                'experiment_id': 'C_stream_sensitivity',
                'model_name': model,
                'batch_size': 32,
                'stream_length': stream_length,
                'n_observations': len(subset),
                'accuracy_mean': subset['metric_accuracy'].mean(),
                'accuracy_std': subset['metric_accuracy'].std(),
                'precision_mean': subset['metric_precision'].mean(),
                'precision_std': subset['metric_precision'].std(),
                'recall_mean': subset['metric_recall'].mean(),
                'recall_std': subset['metric_recall'].std(),
                'f1_mean': subset['metric_f1'].mean(),
                'f1_std': subset['metric_f1'].std(),
                'training_time_mean': subset['model_training_time_seconds'].mean(),
                'training_time_std': subset['model_training_time_seconds'].std(),
                'update_time_mean': subset['batch_update_time_seconds'].mean(),
                'update_time_std': subset['batch_update_time_seconds'].std(),
                'prediction_time_mean': subset['prediction_time_seconds'].mean(),
                'prediction_time_std': subset['prediction_time_seconds'].std(),
                'e2e_time_mean': subset['end_to_end_time_seconds'].mean(),
                'e2e_time_std': subset['end_to_end_time_seconds'].std(),
                'throughput_mean': subset['throughput_records_per_second'].mean(),
                'throughput_std': subset['throughput_records_per_second'].std(),
            })

# Create DataFrame and save
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results/summary_statistics.csv', index=False)

print("Summary Statistics Generated")
print("="*100)
print(summary_df.to_string())
print("\nSaved to: results/summary_statistics.csv")
