"""Visualization functions for experiment results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(results_dir="results/", suffix="raw_results"):
    """Load experiment results from CSV."""
    path = Path(results_dir) / f"{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


def plot_accuracy_f1_by_model(results_df, output_dir="results/", filename="accuracy_f1_by_model.png"):
    """Plot accuracy and F1-score comparison by model for Experiment A (main comparison)."""
    # Filter for Experiment A (base performance)
    exp_a = results_df[results_df['experiment_id'] == 'A_performance'].copy()
    
    if exp_a.empty:
        print("No Experiment A results found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get mean and std for each model
    models = exp_a['model_name'].unique()
    colors = {'classical': '#1f77b4', 'quantum': '#ff7f0e'}
    
    # Accuracy plot
    acc_data = []
    acc_err = []
    labels = []
    bar_colors = []
    
    for model in sorted(models):
        subset = exp_a[exp_a['model_name'] == model]
        acc_mean = subset['metric_accuracy'].mean()
        acc_std = subset['metric_accuracy'].std()
        acc_data.append(acc_mean)
        acc_err.append(acc_std)
        labels.append(model.capitalize())
        bar_colors.append(colors[model])
    
    axes[0].bar(labels, acc_data, yerr=acc_err, capsize=5, color=bar_colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Comparison (Experiment A)', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    for i, (acc, err) in enumerate(zip(acc_data, acc_err)):
        axes[0].text(i, acc + err + 0.02, f'{acc:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    # F1-score plot
    f1_data = []
    f1_err = []
    
    for model in sorted(models):
        subset = exp_a[exp_a['model_name'] == model]
        f1_mean = subset['metric_f1'].mean()
        f1_std = subset['metric_f1'].std()
        f1_data.append(f1_mean)
        f1_err.append(f1_std)
    
    axes[1].bar(labels, f1_data, yerr=f1_err, capsize=5, color=bar_colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1-Score Comparison (Experiment A)', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    for i, (f1, err) in enumerate(zip(f1_data, f1_err)):
        axes[1].text(i, f1 + err + 0.02, f'{f1:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_timing_vs_batch_size(results_df, output_dir="results/", filename="timing_vs_batch_size.png"):
    """Plot training time, update time, and end-to-end time vs batch size."""
    # Filter for Experiment B (batch sensitivity)
    exp_b = results_df[results_df['experiment_id'] == 'B_batch_sensitivity'].copy()
    
    if exp_b.empty:
        print("No Experiment B results found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = sorted(exp_b['model_name'].unique())
    colors = {'classical': '#1f77b4', 'quantum': '#ff7f0e'}
    batch_sizes = sorted(exp_b['batch_size'].unique())
    
    # Plot 1: Training time vs batch size
    for model in models:
        subset = exp_b[exp_b['model_name'] == model]
        data = subset.groupby('batch_size').agg({
            'model_training_time_seconds': ['mean', 'std']
        }).reset_index()
        data.columns = ['batch_size', 'mean', 'std']
        data = data.sort_values('batch_size')
        axes[0].errorbar(data['batch_size'], data['mean'], yerr=data['std'], marker='o', 
                        label=model.capitalize(), color=colors[model], linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Time vs Batch Size', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(batch_sizes)
    
    # Plot 2: Update time vs batch size (classical only, quantum is 0)
    subset = exp_b[exp_b['model_name'] == 'classical']
    data = subset.groupby('batch_size').agg({
        'batch_update_time_seconds': ['mean', 'std']
    }).reset_index()
    data.columns = ['batch_size', 'mean', 'std']
    data = data.sort_values('batch_size')
    axes[1].errorbar(data['batch_size'], data['mean'], yerr=data['std'], marker='s', 
                    label='Classical', color=colors['classical'], linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Update Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Update Time vs Batch Size', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(batch_sizes)
    
    # Plot 3: End-to-end time vs batch size
    for model in models:
        subset = exp_b[exp_b['model_name'] == model]
        data = subset.groupby('batch_size').agg({
            'end_to_end_time_seconds': ['mean', 'std']
        }).reset_index()
        data.columns = ['batch_size', 'mean', 'std']
        data = data.sort_values('batch_size')
        axes[2].errorbar(data['batch_size'], data['mean'], yerr=data['std'], marker='d', 
                        label=model.capitalize(), color=colors[model], linewidth=2, markersize=8)
    
    axes[2].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('End-to-End Time (seconds)', fontsize=12, fontweight='bold')
    axes[2].set_title('End-to-End Time vs Batch Size', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(batch_sizes)
    
    plt.tight_layout()
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_throughput_comparison(results_df, output_dir="results/", filename="throughput_comparison.png"):
    """Plot throughput (records/second) comparison by model for different batch sizes."""
    # Filter for Experiment B
    exp_b = results_df[results_df['experiment_id'] == 'B_batch_sensitivity'].copy()
    
    if exp_b.empty:
        print("No Experiment B results found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = sorted(exp_b['model_name'].unique())
    colors = {'classical': '#1f77b4', 'quantum': '#ff7f0e'}
    batch_sizes = sorted(exp_b['batch_size'].unique())
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    for idx, model in enumerate(models):
        subset = exp_b[exp_b['model_name'] == model]
        data = subset.groupby('batch_size').agg({
            'throughput_records_per_second': ['mean', 'std']
        }).reset_index()
        data.columns = ['batch_size', 'mean', 'std']
        data = data.sort_values('batch_size')
        
        offset = (idx - 0.5) * width
        ax.bar(x + offset, data['mean'], width, yerr=data['std'], capsize=5, 
              label=model.capitalize(), color=colors[model], alpha=0.7, edgecolor='black')
        
        for i, (mean, std) in enumerate(zip(data['mean'], data['std'])):
            ax.text(i + offset, mean + std + 500, f'{mean:.0f}', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (records/second)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Comparison by Batch Size', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_stream_length_sensitivity(results_df, output_dir="results/", filename="stream_length_sensitivity.png"):
    """Plot accuracy and F1 vs stream length for Experiment C."""
    # Filter for Experiment C (stream sensitivity)
    exp_c = results_df[results_df['experiment_id'] == 'C_stream_sensitivity'].copy()
    
    if exp_c.empty:
        print("No Experiment C results found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = sorted(exp_c['model_name'].unique())
    colors = {'classical': '#1f77b4', 'quantum': '#ff7f0e'}
    stream_lengths = sorted(exp_c['stream_length'].unique())
    
    # Plot 1: Accuracy vs stream length
    for model in models:
        subset = exp_c[exp_c['model_name'] == model]
        data = subset.groupby('stream_length').agg({
            'metric_accuracy': ['mean', 'std']
        }).reset_index()
        data.columns = ['stream_length', 'mean', 'std']
        data = data.sort_values('stream_length')
        axes[0].errorbar(data['stream_length'], data['mean'], yerr=data['std'], marker='o', 
                        label=model.capitalize(), color=colors[model], linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Training Stream Length (records)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy vs Training Stream Length', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0.7, 1.0)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(stream_lengths)
    
    # Plot 2: F1-score vs stream length
    for model in models:
        subset = exp_c[exp_c['model_name'] == model]
        data = subset.groupby('stream_length').agg({
            'metric_f1': ['mean', 'std']
        }).reset_index()
        data.columns = ['stream_length', 'mean', 'std']
        data = data.sort_values('stream_length')
        axes[1].errorbar(data['stream_length'], data['mean'], yerr=data['std'], marker='s', 
                        label=model.capitalize(), color=colors[model], linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Training Stream Length (records)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1-Score vs Training Stream Length', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0.7, 1.0)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(stream_lengths)
    
    plt.tight_layout()
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(results_df, output_dir="results/", filename="confusion_matrices.png"):
    """Plot confusion matrices for classical and quantum models (Experiment A)."""
    exp_a = results_df[results_df['experiment_id'] == 'A_performance'].copy()
    
    if exp_a.empty:
        print("No Experiment A results found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for idx, model in enumerate(sorted(exp_a['model_name'].unique())):
        subset = exp_a[exp_a['model_name'] == model]
        # Take first row (all should be identical due to determinism)
        cm_str = subset.iloc[0]['confusion_matrix']
        cm = json.loads(cm_str)
        cm = np.array(cm)
        
        # Normalize for visualization
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = axes[idx].imshow(cm_norm, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(f'{model.capitalize()} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Malignant', 'Benign'])
        axes[idx].set_yticklabels(['Malignant', 'Benign'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = axes[idx].text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2%})',
                                     ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black",
                                     fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=axes[idx], label='Normalized Frequency')
    
    plt.tight_layout()
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_all_plots(results_dir="results/"):
    """Generate all visualization plots from results."""
    print("Loading experiment results...")
    results_df = load_results(results_dir, "raw_results")
    
    print("\nGenerating plots...")
    plot_accuracy_f1_by_model(results_df, results_dir)
    plot_timing_vs_batch_size(results_df, results_dir)
    plot_throughput_comparison(results_df, results_dir)
    plot_stream_length_sensitivity(results_df, results_dir)
    plot_confusion_matrices(results_df, results_dir)
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
