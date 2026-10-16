"""
Plotting: Generate visualizations from experiment results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_accuracy_comparison(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[10, 6] = (10, 6)
) -> None:
    """
    Plot accuracy and F1-score comparison by model.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns: model, accuracy, f1
    output_path : Optional[Path], optional
        Path to save figure
    figsize : Tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results_df['model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    accuracies = [results_df[results_df['model'] == m]['accuracy'].mean() for m in models]
    f1_scores = [results_df[results_df['model'] == m]['f1'].mean() for m in models]
    
    ax.bar(x - width/2, accuracies, width, label='Accuracy')
    ax.bar(x + width/2, f1_scores, width, label='F1-Score')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Accuracy vs F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved accuracy plot to {output_path}")
    
    return fig


def plot_timing_comparison(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[10, 6] = (10, 6)
) -> None:
    """
    Plot timing comparison.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with timing columns
    output_path : Optional[Path], optional
        Path to save figure
    figsize : Tuple, optional
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot timing by batch size if available
    if 'batch_size' in results_df.columns and 'inference_time' in results_df.columns:
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            if 'batch_size' in model_data.columns:
                grouped = model_data.groupby('batch_size')['inference_time'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=model)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Inference Time (s)')
        ax.set_title('Inference Time vs. Batch Size')
    
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved timing plot to {output_path}")
    
    return fig


def generate_plots(
    results_dir: str,
    output_format: str = 'png'
) -> None:
    """
    Generate all plots from results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing results CSV
    output_format : str, optional
        Output format (default: 'png')
    """
    results_path = Path(results_dir) / 'raw_results.csv'
    
    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return
    
    try:
        results_df = pd.read_csv(results_path)
        logger.info(f"Loaded results from {results_path}")
        
        # Generate plots
        plot_accuracy_comparison(
            results_df,
            output_path=Path(results_dir) / f'plot_accuracy.{output_format}'
        )
        
        plot_timing_comparison(
            results_df,
            output_path=Path(results_dir) / f'plot_timing.{output_format}'
        )
        
        logger.info("All plots generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample dataframe
    sample_data = {
        'model': ['Classical', 'Quantum', 'Classical', 'Quantum'],
        'accuracy': [0.92, 0.85, 0.90, 0.87],
        'f1': [0.91, 0.84, 0.89, 0.86],
        'batch_size': [32, 32, 64, 64],
        'inference_time': [0.001, 0.050, 0.002, 0.051],
    }
    
    df = pd.DataFrame(sample_data)
    
    fig1 = plot_accuracy_comparison(df)
    fig2 = plot_timing_comparison(df)
    
    print("Plotting smoke test passed.")
