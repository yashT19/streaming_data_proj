"""
Stream simulator: Deterministic batch-oriented streaming for training data.

This module handles:
1. Configurable batch sizes
2. Deterministic shuffling
3. Incomplete final batch handling
4. Record count tracking
5. No data leakage (training data only)
"""

import numpy as np
from typing import Iterator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StreamSimulator:
    """
    Deterministic stream simulator for training data.
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32,
        random_state: int = 42
    ):
        """
        Initialize stream simulator.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        batch_size : int, optional
            Batch size (default: 32)
        random_state : int, optional
            Random state for shuffling (default: 42)
        """
        if batch_size <= 0:
            raise ValueError(f"Invalid batch size: {batch_size}")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Mismatched lengths: {X_train.shape[0]} features vs {y_train.shape[0]} targets")
        
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_records = X_train.shape[0]
        self.n_batches = (self.n_records + batch_size - 1) // batch_size
        
        logger.info(f"StreamSimulator: {self.n_records} records, "
                   f"batch size {batch_size}, {self.n_batches} batches")
    
    def stream_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray, dict]]:
        """
        Yield deterministic batches from training data.
        
        Yields
        ------
        Tuple[np.ndarray, np.ndarray, dict]
            X_batch, y_batch, metadata dict
        """
        # Deterministic shuffle
        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(self.n_records)
        X_shuffled = self.X_train[indices]
        y_shuffled = self.y_train[indices]
        
        cumulative_records = 0
        
        for batch_idx in range(self.n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.n_records)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            batch_records = X_batch.shape[0]
            cumulative_records += batch_records
            
            metadata = {
                'batch_index': batch_idx,
                'batch_size': batch_records,
                'cumulative_records': cumulative_records,
                'is_final_batch': (batch_idx == self.n_batches - 1),
            }
            
            yield X_batch, y_batch, metadata
    
    def validate_stream(self) -> bool:
        """
        Validate that stream emits expected number of records.
        
        Returns
        -------
        bool
            True if validation passes
        """
        total_records = 0
        for X_batch, y_batch, metadata in self.stream_batches():
            total_records += X_batch.shape[0]
        
        if total_records != self.n_records:
            raise ValueError(f"Stream emitted {total_records} records, expected {self.n_records}")
        
        logger.info(f"Stream validation passed: {total_records} records emitted")
        return True


def stream_batches(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 32,
    random_state: int = 42
) -> Iterator[Tuple[np.ndarray, np.ndarray, dict]]:
    """
    Convenience function to stream batches without creating a StreamSimulator object.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    batch_size : int, optional
        Batch size (default: 32)
    random_state : int, optional
        Random state for shuffling (default: 42)
    
    Yields
    ------
    Tuple[np.ndarray, np.ndarray, dict]
        X_batch, y_batch, metadata dict
    """
    simulator = StreamSimulator(X_train, y_train, batch_size, random_state)
    yield from simulator.stream_batches()


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    from data_processing import load_dataset, split_dataset
    
    X, y = load_dataset()
    X_train, _, _, y_train, _, _ = split_dataset(X, y)
    
    simulator = StreamSimulator(X_train, y_train, batch_size=16, random_state=42)
    simulator.validate_stream()
    
    print("\nFirst 3 batches:")
    for i, (X_batch, y_batch, metadata) in enumerate(simulator.stream_batches()):
        if i < 3:
            print(f"Batch {metadata['batch_index']}: {X_batch.shape[0]} records, "
                  f"cumulative {metadata['cumulative_records']}")
    
    print("Stream simulator smoke test passed.")
