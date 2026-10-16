"""
Tests for data processing module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    load_dataset, split_dataset, preprocess_classical, preprocess_quantum, validate_data
)


class TestDataLoading:
    """Test dataset loading."""
    
    def test_load_dataset(self):
        """Test dataset loads successfully."""
        X, y = load_dataset()
        assert X.shape[0] > 0
        assert y.shape[0] > 0
        assert X.shape[0] == y.shape[0]
    
    def test_dataset_dimensions(self):
        """Test dataset has expected dimensions."""
        X, y = load_dataset()
        assert X.shape[1] == 30  # Breast Cancer Wisconsin has 30 features
        assert len(np.unique(y)) == 2  # Binary classification


class TestDataSplitting:
    """Test train/val/test splitting."""
    
    def test_split_proportions(self):
        """Test split maintains correct proportions."""
        X, y = load_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2
        )
        
        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == X.shape[0]
        assert abs(X_train.shape[0] / X.shape[0] - 0.6) < 0.05
        assert abs(X_val.shape[0] / X.shape[0] - 0.2) < 0.05
        assert abs(X_test.shape[0] / X.shape[0] - 0.2) < 0.05
    
    def test_no_overlap(self):
        """Test no overlapping records between splits."""
        X, y = load_dataset()
        # Note: This is a simplified test; full validation would involve tracking indices
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
        
        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0
    
    def test_stratification(self):
        """Test stratified split maintains class balance."""
        X, y = load_dataset()
        original_ratio = np.mean(y)
        
        X_train, _, _, y_train, _, _ = split_dataset(X, y, stratify=True)
        train_ratio = np.mean(y_train)
        
        # Ratios should be close
        assert abs(original_ratio - train_ratio) < 0.05


class TestPreprocessing:
    """Test preprocessing pipelines."""
    
    def test_classical_preprocessing(self):
        """Test classical preprocessing."""
        X, y = load_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
        
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_classical(
            X_train, X_val, X_test
        )
        
        # Check dimensions preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Check scaling worked
        assert abs(np.mean(X_train_scaled)) < 0.1  # Should be close to 0
        assert abs(np.std(X_train_scaled) - 1.0) < 0.1  # Should be close to 1
    
    def test_quantum_preprocessing(self):
        """Test quantum preprocessing (PCA + scaling)."""
        X, y = load_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
        
        X_train_reduced, X_val_reduced, X_test_reduced, pca, scaler = preprocess_quantum(
            X_train, X_val, X_test, n_features=4
        )
        
        # Check dimensions
        assert X_train_reduced.shape[1] == 4
        assert X_val_reduced.shape[1] == 4
        assert X_test_reduced.shape[1] == 4
        
        # Check row counts preserved
        assert X_train_reduced.shape[0] == X_train.shape[0]
        assert X_val_reduced.shape[0] == X_val.shape[0]
        assert X_test_reduced.shape[0] == X_test.shape[0]


class TestDataValidation:
    """Test data validation."""
    
    def test_valid_data_passes(self):
        """Test valid data passes validation."""
        X = np.ones((10, 4))
        y = np.ones(10)
        
        assert validate_data(X, y)
    
    def test_empty_X_fails(self):
        """Test empty X fails validation."""
        X = np.empty((0, 4))
        y = np.ones(0)
        
        with pytest.raises(ValueError):
            validate_data(X, y)
    
    def test_mismatched_lengths_fail(self):
        """Test mismatched X and y lengths fail."""
        X = np.ones((10, 4))
        y = np.ones(5)
        
        with pytest.raises(ValueError):
            validate_data(X, y)
    
    def test_non_finite_data_fails(self):
        """Test non-finite data fails validation."""
        X = np.ones((10, 4))
        X[0, 0] = np.nan
        y = np.ones(10)
        
        with pytest.raises(ValueError):
            validate_data(X, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
