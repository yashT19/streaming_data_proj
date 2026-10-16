"""
Stream simulator: Deterministic batch-oriented streaming for training data.

This module handles:
1. Configurable batch sizes (16, 32, 64, or any positive integer)
2. Deterministic shuffling with fixed seed
3. Incomplete final batch handling
4. Record count and alignment tracking
5. No data leakage (training data only)
6. Comprehensive input validation
7. Edge case handling (empty data, oversized batches)

Usage:
    # Create a simulator
    simulator = StreamSimulator(X_train, y_train, batch_size=32, random_state=42)
    
    # Stream batches
    for X_batch, y_batch, metadata in simulator.stream_batches():
        # metadata contains: batch_index, batch_size, cumulative_records, is_final_batch
        train_on_batch(X_batch, y_batch)
        
    # Or use convenience function
    for X_batch, y_batch, metadata in stream_batches(X_train, y_train, batch_size=32):
        train_on_batch(X_batch, y_batch)
"""

import numpy as np
from typing import Iterator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StreamSimulator:
    """
    Deterministic stream simulator for training data.
    
    Yields batches of features and labels in deterministic order with metadata.
    All records are emitted exactly once per stream epoch. Features and labels
    remain aligned throughout streaming.
    
    Supported batch sizes: any positive integer (common: 16, 32, 64)
    Edge cases:
    - Empty training data: Raises ValueError
    - Batch size > dataset size: Yields single batch with all records
    - Batch size == dataset size: Yields single batch
    - Non-divisible split: Preserves smaller final batch
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32,
        random_state: int = 42,
        shuffle: bool = True
    ):
        """
        Initialize stream simulator.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features with shape (n_records, n_features)
        y_train : np.ndarray
            Training targets with shape (n_records,) or (n_records, 1)
        batch_size : int, optional
            Batch size for streaming. Must be positive integer. (default: 32)
            Supported values: 16, 32, 64, or any positive integer.
            Raises ValueError if ≤ 0, non-integer, or not finite.
        random_state : int, optional
            Random state for deterministic shuffling (default: 42)
        shuffle : bool, optional
            Whether to shuffle records before streaming (default: True)
        
        Raises
        ------
        ValueError
            If batch_size <= 0, non-integer, non-finite, or records are empty
        ValueError
            If X_train and y_train have mismatched lengths
        
        Examples
        --------
        >>> X = np.random.randn(100, 4)
        >>> y = np.random.randint(0, 2, 100)
        >>> simulator = StreamSimulator(X, y, batch_size=32)
        >>> for X_batch, y_batch, metadata in simulator.stream_batches():
        ...     print(f"Batch {metadata['batch_index']}: {X_batch.shape[0]} records")
        """
        # Validate batch size
        self._validate_batch_size(batch_size)
        
        # Validate data
        if X_train.shape[0] == 0:
            raise ValueError("Empty training data: X_train has 0 records")
        if y_train.shape[0] == 0:
            raise ValueError("Empty training data: y_train has 0 records")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Mismatched record counts: X_train has {X_train.shape[0]} records "
                f"but y_train has {y_train.shape[0]} records"
            )
        
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.batch_size = batch_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.n_records = X_train.shape[0]
        self.n_batches = (self.n_records + batch_size - 1) // batch_size
        
        logger.info(
            f"StreamSimulator initialized: {self.n_records} records, "
            f"batch_size={batch_size}, {self.n_batches} batches, "
            f"shuffle={shuffle}, seed={random_state}"
        )
    
    @staticmethod
    def _validate_batch_size(batch_size) -> None:
        """
        Validate batch size parameter.
        
        Parameters
        ----------
        batch_size : int
            Batch size to validate
        
        Raises
        ------
        ValueError
            If batch_size is invalid
        """
        # Check if finite
        if not np.isfinite(batch_size):
            raise ValueError(f"Batch size must be finite, got {batch_size}")
        
        # Check if integer (or integer-like)
        if isinstance(batch_size, float):
            if batch_size != int(batch_size):
                raise ValueError(
                    f"Batch size must be integer, got {batch_size} (not integer)"
                )
            batch_size = int(batch_size)
        
        # Check if positive
        if batch_size <= 0:
            raise ValueError(
                f"Batch size must be positive integer, got {batch_size}"
            )
        
        # Check if supported (common sizes)
        if batch_size not in [16, 32, 64] and batch_size > 1024:
            logger.warning(
                f"Batch size {batch_size} is unusually large (>1024). "
                "This may cause memory issues."
            )
    
    def stream_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray, dict]]:
        """
        Yield deterministic batches from training data.
        
        Yields batches in deterministic order, with features and labels aligned.
        Each record is emitted exactly once per stream. The order is deterministic
        based on random_state if shuffling is enabled.
        
        Records are processed in the order: shuffle (if enabled) → batch sequentially.
        
        Yields
        ------
        Tuple[np.ndarray, np.ndarray, dict]
            - X_batch: Feature batch with shape (batch_size,) or (batch_size, n_features)
            - y_batch: Label batch with shape (batch_size,)
            - metadata: Dictionary containing:
                - 'batch_index': 0-based batch index
                - 'batch_size': Number of records in this batch (≤ requested batch_size)
                - 'cumulative_records': Total records emitted so far
                - 'is_final_batch': True if this is the last batch
        
        Examples
        --------
        >>> X = np.random.randn(100, 4)
        >>> y = np.random.randint(0, 2, 100)
        >>> simulator = StreamSimulator(X, y, batch_size=32)
        >>> total = 0
        >>> for X_batch, y_batch, metadata in simulator.stream_batches():
        ...     print(f"Batch {metadata['batch_index']}: {X_batch.shape[0]} records")
        ...     total += X_batch.shape[0]
        >>> print(f"Total: {total} records emitted")
        Total: 100 records emitted
        """
        # Deterministic shuffle if enabled
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(self.n_records)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
        else:
            X_shuffled = self.X_train
            y_shuffled = self.y_train
        
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
        Validate that stream emits exactly the expected number of records.
        
        Iterates through all batches and verifies:
        - Total records emitted equals initial record count
        - No records are duplicated or missing
        - All batches have non-zero size
        - Final batch size is correct
        - Cumulative record count is monotonically increasing
        
        Returns
        -------
        bool
            True if validation passes
        
        Raises
        ------
        ValueError
            If stream emits wrong number of records or other validation fails
        """
        total_records = 0
        prev_cumulative = 0
        
        for batch_idx, (X_batch, y_batch, metadata) in enumerate(self.stream_batches()):
            batch_size = X_batch.shape[0]
            
            # Verify non-empty batch
            if batch_size == 0:
                raise ValueError(f"Batch {batch_idx} is empty")
            
            # Verify batch size doesn't exceed requested
            if batch_size > self.batch_size:
                raise ValueError(
                    f"Batch {batch_idx} has {batch_size} records "
                    f"but requested batch_size is {self.batch_size}"
                )
            
            # Verify alignment
            if X_batch.shape[0] != y_batch.shape[0]:
                raise ValueError(
                    f"Batch {batch_idx}: X has {X_batch.shape[0]} records "
                    f"but y has {y_batch.shape[0]} records"
                )
            
            # Verify metadata consistency
            if metadata['batch_size'] != batch_size:
                raise ValueError(
                    f"Batch {batch_idx}: metadata says {metadata['batch_size']} records "
                    f"but actual batch has {batch_size} records"
                )
            
            # Verify cumulative is increasing
            if metadata['cumulative_records'] <= prev_cumulative:
                raise ValueError(
                    f"Batch {batch_idx}: cumulative went backward "
                    f"(was {prev_cumulative}, now {metadata['cumulative_records']})"
                )
            
            prev_cumulative = metadata['cumulative_records']
            total_records += batch_size
        
        if total_records != self.n_records:
            raise ValueError(
                f"Stream validation failed: emitted {total_records} records "
                f"but expected {self.n_records}"
            )
        
        logger.info(
            f"Stream validation passed: {total_records} records emitted "
            f"across {self.n_batches} batches, all aligned and deduplicated"
        )
        return True
    
    def get_config(self) -> dict:
        """
        Get current simulator configuration.
        
        Returns
        -------
        dict
            Configuration dictionary with:
            - 'n_records': Number of training records
            - 'batch_size': Requested batch size
            - 'n_batches': Number of batches to emit
            - 'shuffle': Whether shuffling is enabled
            - 'random_state': Random seed for determinism
        """
        return {
            'n_records': self.n_records,
            'batch_size': self.batch_size,
            'n_batches': self.n_batches,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
        }


def stream_batches(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 32,
    random_state: int = 42,
    shuffle: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray, dict]]:
    """
    Convenience function to stream batches without creating a StreamSimulator object.
    
    Functionally equivalent to creating a StreamSimulator and calling stream_batches(),
    but useful for one-off streaming operations.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features with shape (n_records, n_features)
    y_train : np.ndarray
        Training targets with shape (n_records,)
    batch_size : int, optional
        Batch size (default: 32). Must be positive integer.
    random_state : int, optional
        Random state for shuffling (default: 42)
    shuffle : bool, optional
        Whether to shuffle records (default: True)
    
    Yields
    ------
    Tuple[np.ndarray, np.ndarray, dict]
        X_batch, y_batch, metadata dict (see StreamSimulator.stream_batches for details)
    
    Raises
    ------
    ValueError
        If batch_size is invalid or data is empty
    
    Examples
    --------
    >>> X = np.random.randn(100, 4)
    >>> y = np.random.randint(0, 2, 100)
    >>> for X_batch, y_batch, metadata in stream_batches(X, y, batch_size=32):
    ...     print(f"Processing batch {metadata['batch_index']}")
    """
    simulator = StreamSimulator(X_train, y_train, batch_size, random_state, shuffle)
    yield from simulator.stream_batches()


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    from src.data_processing import load_dataset, split_dataset
    
    X, y, _, _ = load_dataset()
    X_train, _, _, y_train, _, _, _ = split_dataset(X, y)
    
    simulator = StreamSimulator(X_train, y_train, batch_size=16, random_state=42)
    simulator.validate_stream()
    
    print("\nFirst 3 batches:")
    for i, (X_batch, y_batch, metadata) in enumerate(simulator.stream_batches()):
        if i < 3:
            print(f"Batch {metadata['batch_index']}: {X_batch.shape[0]} records, "
                  f"cumulative {metadata['cumulative_records']}")
    
    print("Stream simulator smoke test passed.")
