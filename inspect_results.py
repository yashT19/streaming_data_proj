#!/usr/bin/env python
"""Inspect raw results structure and data quality."""
import pandas as pd
import numpy as np
import json

# Load and inspect raw results
raw = pd.read_csv('results/raw_results.csv')

print("="*90)
print("RAW RESULTS INSPECTION")
print("="*90)

print(f"\nShape: {raw.shape}")
print(f"Columns: {len(raw.columns)}")
print(f"\nFirst few columns:\n{raw.columns[:10].tolist()}")

print(f"\n\nStatus Distribution:")
print(raw['status'].value_counts())

print(f"\n\nExperiment ID Distribution:")
print(raw['experiment_id'].value_counts())

print(f"\n\nModel Distribution:")
print(raw['model_name'].value_counts())

print(f"\n\nSample row (first observation):")
for col in ['experiment_id', 'model_name', 'batch_size', 'stream_length', 'metric_accuracy', 'metric_f1', 'model_training_time_seconds', 'end_to_end_time_seconds']:
    if col in raw.columns:
        print(f"  {col}: {raw[col].iloc[0]}")

print(f"\n\nMetrics summary (successful runs only):")
successful = raw[raw['status'] == 'success']
for col in ['metric_accuracy', 'metric_precision', 'metric_recall', 'metric_f1']:
    if col in successful.columns:
        vals = successful[col].dropna()
        print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

print(f"\n\nTiming summary:")
for col in ['model_training_time_seconds', 'batch_update_time_seconds', 'prediction_time_seconds', 'end_to_end_time_seconds']:
    if col in successful.columns:
        vals = successful[col].dropna()
        print(f"  {col}: min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}")

print("\n" + "="*90)
