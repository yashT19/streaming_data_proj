"""
Tests for stream simulator.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stream_simulator import StreamSimulator, stream_batches


class TestStreamSimulator:
    """Test streaming functionality."""
    
    def test_initialization(self):
        """Test StreamSimulator initialization."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        sim = StreamSimulator(X_train, y_train, batch_size=32)
        
        assert sim.n_records == 100
        assert sim.batch_size == 32
        assert sim.n_batches == 4  # 100 / 32 = 3.125 → 4 batches
    
    def test_invalid_batch_size(self):
        """Test invalid batch size raises error."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        with pytest.raises(ValueError):
            StreamSimulator(X_train, y_train, batch_size=0)
    
    def test_mismatched_lengths(self):
        """Test mismatched X and y lengths raise error."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 50)
        
        with pytest.raises(ValueError):
            StreamSimulator(X_train, y_train, batch_size=32)
    
    def test_batch_streaming(self):
        """Test batches are yielded correctly."""
        X_train = np.arange(100).reshape((100, 1))
        y_train = np.arange(100)
        
        sim = StreamSimulator(X_train, y_train, batch_size=32)
        
        batches = list(sim.stream_batches())
        assert len(batches) == 4  # Expected number of batches
        
        # Check batch sizes
        assert batches[0][0].shape[0] == 32
        assert batches[1][0].shape[0] == 32
        assert batches[2][0].shape[0] == 32
        assert batches[3][0].shape[0] == 4  # Incomplete final batch
    
    def test_record_count(self):
        """Test all records are streamed exactly once."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        sim = StreamSimulator(X_train, y_train, batch_size=32)
        
        total_records = 0
        for X_batch, y_batch, metadata in sim.stream_batches():
            total_records += X_batch.shape[0]
        
        assert total_records == 100
    
    def test_metadata(self):
        """Test batch metadata is correct."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        sim = StreamSimulator(X_train, y_train, batch_size=32)
        
        batches = list(sim.stream_batches())
        
        # Check first batch
        assert batches[0][2]['batch_index'] == 0
        assert batches[0][2]['batch_size'] == 32
        assert batches[0][2]['is_final_batch'] == False
        
        # Check final batch
        assert batches[-1][2]['is_final_batch'] == True
    
    def test_stream_validation(self):
        """Test stream validation."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        sim = StreamSimulator(X_train, y_train, batch_size=32)
        
        # Should not raise
        assert sim.validate_stream() == True
    
    def test_edge_case_single_batch(self):
        """Test edge case: single batch."""
        X_train = np.random.randn(20, 4)
        y_train = np.random.randint(0, 2, 20)
        
        sim = StreamSimulator(X_train, y_train, batch_size=32)
        
        batches = list(sim.stream_batches())
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 20
    
    def test_edge_case_batch_equals_n(self):
        """Test edge case: batch size equals number of records."""
        X_train = np.random.randn(32, 4)
        y_train = np.random.randint(0, 2, 32)
        
        sim = StreamSimulator(X_train, y_train, batch_size=32)
        
        batches = list(sim.stream_batches())
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 32
    
    def test_deterministic(self):
        """Test stream is deterministic with same seed."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        sim1 = StreamSimulator(X_train, y_train, batch_size=32, random_state=42)
        batches1 = list(sim1.stream_batches())
        
        sim2 = StreamSimulator(X_train, y_train, batch_size=32, random_state=42)
        batches2 = list(sim2.stream_batches())
        
        for b1, b2 in zip(batches1, batches2):
            np.testing.assert_array_equal(b1[0], b2[0])
            np.testing.assert_array_equal(b1[1], b2[1])


class TestStreamBatchesFunction:
    """Test convenience stream_batches function."""
    
    def test_stream_batches_function(self):
        """Test stream_batches() convenience function."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        batches = list(stream_batches(X_train, y_train, batch_size=32))
        
        assert len(batches) == 4
        assert batches[0][0].shape[0] == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
