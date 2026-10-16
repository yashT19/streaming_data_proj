#!/usr/bin/env python
"""Generate publication-quality plots from actual experiment results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
RESULTS_DIR = Path('results')
PLOTS_DIR = RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading experiment data...")
raw = pd.read_csv(RESULTS_DIR / 'raw_results.csv')
successful = raw[raw['status'] == 'success'].copy()

print(f"Loaded {len(successful)} successful observations")
print(f"Experiments: {successful['experiment_id'].unique().tolist()}")

# Set matplotlib style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10

# Color scheme
COLORS = {'classical': '#1f77b4', 'quantum': '#ff7f0e'}

# ============================================================================
# FIGURE 1: Accuracy and F1-Score Comparison
# ============================================================================
print("\n[Figure 1/5] Accuracy and F1-Score Comparison...")

exp_a = successful[successful['experiment_id'] == 'A_performance']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
models = ['classical', 'quantum']
acc_means = []
acc_stds = []
for model in models:
    subset = exp_a[exp_a['model_name'] == model]
    acc_means.append(subset['metric_accuracy'].mean())
    acc_stds.append(subset['metric_accuracy'].std())

x_pos = np.arange(len(models))
axes[0].bar(x_pos, acc_means, yerr=acc_stds, capsize=5, 
            color=[COLORS[m] for m in models], alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
axes[0].set_title('Predictive Accuracy Comparison\n(Experiment A: Full Training Stream, Batch Size 32)', 
                   fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([m.capitalize() for m in models])
axes[0].set_ylim([0.6, 1.0])
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
for i, (mean, std) in enumerate(zip(acc_means, acc_stds)):
    axes[0].text(i, mean + std + 0.02, f'{mean:.4f}', ha='center', fontweight='bold', fontsize=10)

# F1-Score
f1_means = []
f1_stds = []
for model in models:
    subset = exp_a[exp_a['model_name'] == model]
    f1_means.append(subset['metric_f1'].mean())
    f1_stds.append(subset['metric_f1'].std())

axes[1].bar(x_pos, f1_means, yerr=f1_stds, capsize=5,
            color=[COLORS[m] for m in models], alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
axes[1].set_title('F1-Score Comparison\n(Experiment A: Full Training Stream, Batch Size 32)',
                   fontsize=12, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([m.capitalize() for m in models])
axes[1].set_ylim([0.7, 1.0])
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
for i, (mean, std) in enumerate(zip(f1_means, f1_stds)):
    axes[1].text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_accuracy_f1_comparison.png', bbox_inches='tight')
print(f"    ✓ Saved: {PLOTS_DIR / '01_accuracy_f1_comparison.png'}")
plt.close()

# ============================================================================
# FIGURE 2: Processing Time vs Batch Size
# ============================================================================
print("[Figure 2/5] Processing Time vs Batch Size...")

exp_b = successful[successful['experiment_id'] == 'B_batch_sensitivity']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

batch_sizes = sorted(exp_b['batch_size'].unique())

# Training time
for model in models:
    times_by_batch = []
    stds_by_batch = []
    for batch_size in batch_sizes:
        subset = exp_b[(exp_b['batch_size'] == batch_size) & (exp_b['model_name'] == model)]
        times_by_batch.append(subset['model_training_time_seconds'].mean())
        stds_by_batch.append(subset['model_training_time_seconds'].std())
    axes[0].errorbar(batch_sizes, times_by_batch, yerr=stds_by_batch, marker='o', 
                     label=model.capitalize(), color=COLORS[model], linewidth=2, markersize=8, capsize=5)

axes[0].set_xlabel('Batch Size (records)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
axes[0].set_title('Training Time vs Batch Size', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].set_xticks(batch_sizes)

# Update/Inference time (combined)
for model in models:
    times_by_batch = []
    stds_by_batch = []
    for batch_size in batch_sizes:
        subset = exp_b[(exp_b['batch_size'] == batch_size) & (exp_b['model_name'] == model)]
        times_by_batch.append(subset['prediction_time_seconds'].mean())
        stds_by_batch.append(subset['prediction_time_seconds'].std())
    axes[1].errorbar(batch_sizes, times_by_batch, yerr=stds_by_batch, marker='s',
                     label=model.capitalize(), color=COLORS[model], linewidth=2, markersize=8, capsize=5)

axes[1].set_xlabel('Batch Size (records)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Prediction Time (seconds)', fontsize=11, fontweight='bold')
axes[1].set_title('Prediction Time vs Batch Size', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].set_xticks(batch_sizes)

# End-to-end time
for model in models:
    times_by_batch = []
    stds_by_batch = []
    for batch_size in batch_sizes:
        subset = exp_b[(exp_b['batch_size'] == batch_size) & (exp_b['model_name'] == model)]
        times_by_batch.append(subset['end_to_end_time_seconds'].mean())
        stds_by_batch.append(subset['end_to_end_time_seconds'].std())
    axes[2].errorbar(batch_sizes, times_by_batch, yerr=stds_by_batch, marker='^',
                     label=model.capitalize(), color=COLORS[model], linewidth=2, markersize=8, capsize=5)

axes[2].set_xlabel('Batch Size (records)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('End-to-End Time (seconds)', fontsize=11, fontweight='bold')
axes[2].set_title('End-to-End Time vs Batch Size', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(alpha=0.3, linestyle='--')
axes[2].set_xticks(batch_sizes)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_timing_vs_batch_size.png', bbox_inches='tight')
print(f"    ✓ Saved: {PLOTS_DIR / '02_timing_vs_batch_size.png'}")
plt.close()

# ============================================================================
# FIGURE 3: Throughput Comparison
# ============================================================================
print("[Figure 3/5] Throughput Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(batch_sizes))
width = 0.35

for idx, model in enumerate(models):
    throughputs = []
    stds = []
    for batch_size in batch_sizes:
        subset = exp_b[(exp_b['batch_size'] == batch_size) & (exp_b['model_name'] == model)]
        throughputs.append(subset['throughput_records_per_second'].mean())
        stds.append(subset['throughput_records_per_second'].std())
    
    offset = (idx - 0.5) * width
    ax.bar(x + offset, throughputs, width, yerr=stds, capsize=5,
           label=model.capitalize(), color=COLORS[model], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (tp, std) in enumerate(zip(throughputs, stds)):
        ax.text(i + offset, tp + std + 1000, f'{tp:.0f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Batch Size (records)', fontsize=11, fontweight='bold')
ax.set_ylabel('Throughput (records/second)', fontsize=11, fontweight='bold')
ax.set_title('Throughput Comparison by Batch Size\n(Training + Prediction Phase)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_throughput_comparison.png', bbox_inches='tight')
print(f"    ✓ Saved: {PLOTS_DIR / '03_throughput_comparison.png'}")
plt.close()

# ============================================================================
# FIGURE 4: Stream Length Sensitivity
# ============================================================================
print("[Figure 4/5] Stream Length Sensitivity...")

exp_c = successful[successful['experiment_id'] == 'C_stream_sensitivity']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stream_lengths = sorted(exp_c['stream_length'].unique())

# Accuracy vs stream length
for model in models:
    accs = []
    stds = []
    for stream_len in stream_lengths:
        subset = exp_c[(exp_c['stream_length'] == stream_len) & (exp_c['model_name'] == model)]
        accs.append(subset['metric_accuracy'].mean())
        stds.append(subset['metric_accuracy'].std())
    axes[0].errorbar(stream_lengths, accs, yerr=stds, marker='o', label=model.capitalize(),
                     color=COLORS[model], linewidth=2, markersize=8, capsize=5)

axes[0].set_xlabel('Training Stream Length (records)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
axes[0].set_title('Accuracy vs Training Stream Length\n(Batch Size 32)', fontsize=12, fontweight='bold')
axes[0].set_ylim([0.7, 1.0])
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].set_xticks(stream_lengths)

# F1 vs stream length
for model in models:
    f1s = []
    stds = []
    for stream_len in stream_lengths:
        subset = exp_c[(exp_c['stream_length'] == stream_len) & (exp_c['model_name'] == model)]
        f1s.append(subset['metric_f1'].mean())
        stds.append(subset['metric_f1'].std())
    axes[1].errorbar(stream_lengths, f1s, yerr=stds, marker='s', label=model.capitalize(),
                     color=COLORS[model], linewidth=2, markersize=8, capsize=5)

axes[1].set_xlabel('Training Stream Length (records)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
axes[1].set_title('F1-Score vs Training Stream Length\n(Batch Size 32)', fontsize=12, fontweight='bold')
axes[1].set_ylim([0.7, 1.0])
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].set_xticks(stream_lengths)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_stream_length_sensitivity.png', bbox_inches='tight')
print(f"    ✓ Saved: {PLOTS_DIR / '04_stream_length_sensitivity.png'}")
plt.close()

# ============================================================================
# FIGURE 5: Update/Training Cost Comparison
# ============================================================================
print("[Figure 5/5] Update/Training Cost Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Classical: Update cost per batch size
exp_b_classical = exp_b[exp_b['model_name'] == 'classical']
update_by_batch = []
update_std = []
for batch_size in batch_sizes:
    subset = exp_b_classical[exp_b_classical['batch_size'] == batch_size]
    update_by_batch.append(subset['batch_update_time_seconds'].mean())
    update_std.append(subset['batch_update_time_seconds'].std())

axes[0].bar(batch_sizes, update_by_batch, yerr=update_std, capsize=5,
            color=COLORS['classical'], alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Batch Size (records)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Batch Update Time (seconds)', fontsize=11, fontweight='bold')
axes[0].set_title('Classical: Batch Update Cost vs Batch Size', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].set_xticks(batch_sizes)
for i, (time, std) in enumerate(zip(update_by_batch, update_std)):
    axes[0].text(batch_sizes[i], time + std + 0.0005, f'{time*1000:.2f}ms', ha='center', fontsize=9, fontweight='bold')

# Quantum: Training cost
exp_a_quantum = exp_a[exp_a['model_name'] == 'quantum']
train_quantum = exp_a_quantum['model_training_time_seconds'].mean()
train_quantum_std = exp_a_quantum['model_training_time_seconds'].std()

exp_a_classical = exp_a[exp_a['model_name'] == 'classical']
train_classical = exp_a_classical['model_training_time_seconds'].mean()
train_classical_std = exp_a_classical['model_training_time_seconds'].std()

models_list = ['Classical\n(Experiment A)', 'Quantum\n(Experiment A)']
train_costs = [train_classical, train_quantum]
train_stds = [train_classical_std, train_quantum_std]
colors_list = [COLORS['classical'], COLORS['quantum']]

axes[1].bar(models_list, train_costs, yerr=train_stds, capsize=5,
            color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
axes[1].set_title('Training Cost Comparison\n(Full Training Stream, Batch Size 32)', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
for i, (cost, std) in enumerate(zip(train_costs, train_stds)):
    axes[1].text(i, cost + std + 0.2, f'{cost:.3f}s', ha='center', fontsize=10, fontweight='bold')

speedup_ratio = train_quantum / train_classical
axes[1].text(0.5, max(train_costs) * 0.5, f'{speedup_ratio:.0f}× speedup\n(Classical faster)', 
             ha='center', fontsize=11, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_training_cost_comparison.png', bbox_inches='tight')
print(f"    ✓ Saved: {PLOTS_DIR / '05_training_cost_comparison.png'}")
plt.close()

print("\n" + "="*80)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("="*80)
print(f"\nFigures saved to: {PLOTS_DIR.absolute()}")
print("\nGenerated figures:")
for i, fig in enumerate(sorted(PLOTS_DIR.glob('*.png')), 1):
    print(f"  {i}. {fig.name}")
