"""
Experiment runner: Orchestrate all experiments and save results.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import sys

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load experiment configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    
    Returns
    -------
    Dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def save_results(
    results: List[Dict],
    output_dir: str,
    format: str = 'csv'
) -> None:
    """
    Save raw results to file.
    
    Parameters
    ----------
    results : List[Dict]
        List of result dictionaries
    output_dir : str
        Output directory
    format : str, optional
        Output format: 'csv' or 'json' (default: 'csv')
    """
    output_path = Path(output_dir) / f'raw_results.{format}'
    
    if format == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    elif format == 'json':
        df = pd.DataFrame(results)
        df.to_json(output_path, orient='records', indent=2)
    
    logger.info(f"Saved {len(results)} results to {output_path}")


def save_summary(
    results: List[Dict],
    output_dir: str,
    format: str = 'csv'
) -> None:
    """
    Save aggregated summary statistics.
    
    Parameters
    ----------
    results : List[Dict]
        List of result dictionaries
    output_dir : str
        Output directory
    format : str, optional
        Output format: 'csv' or 'json' (default: 'csv')
    """
    output_path = Path(output_dir) / f'summary.{format}'
    
    df = pd.DataFrame(results)
    
    # Group by model and compute statistics
    summary = df.groupby('model')[['accuracy', 'precision', 'recall', 'f1']].agg(['mean', 'std'])
    
    if format == 'csv':
        summary.to_csv(output_path)
    elif format == 'json':
        summary.to_json(output_path, indent=2)
    
    logger.info(f"Saved summary to {output_path}")


def run_smoke_test() -> None:
    """
    Run a quick smoke test to validate the pipeline.
    """
    logger.info("Starting smoke test...")
    
    try:
        from src.data_processing import load_dataset, split_dataset, preprocess_classical, preprocess_quantum
        from src.stream_simulator import StreamSimulator
        from src.classical_model import ClassicalModel
        
        # Load and preprocess data
        X, y = load_dataset()
        X_train, _, X_test, y_train, _, y_test = split_dataset(X, y)
        X_train_c, _, X_test_c, _ = preprocess_classical(X_train, X_train, X_test)
        
        # Stream simulator
        simulator = StreamSimulator(X_train_c, y_train, batch_size=32)
        simulator.validate_stream()
        
        # Classical model
        model = ClassicalModel()
        model.train(X_train_c[:100], y_train[:100])
        y_pred, timing = model.predict(X_test_c)
        
        logger.info("Smoke test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        return False


def run_full_experiment() -> None:
    """
    Run the full experimental pipeline.
    """
    logger.info("Starting full experiment...")
    
    # This will be implemented in Phase 6
    logger.info("Full experiment implementation pending")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_smoke_test()
