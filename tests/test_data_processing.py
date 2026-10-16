"""
Tests for data processing module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    load_dataset, split_dataset, preprocess_classical, preprocess_quantum, validate_data,
    check_split_overlap, report_dataset_statistics, prepare_data
)


class TestDataLoading:
    """Test dataset loading."""
    
    def test_load_dataset(self):
        """Test dataset loads successfully."""
        X, y, feature_names, target_names = load_dataset()
        assert X.shape[0] > 0
        assert y.shape[0] > 0
        assert X.shape[0] == y.shape[0]
        assert feature_names is not None
        assert target_names is not None
    
    def test_dataset_dimensions(self):
        """Test dataset has expected dimensions."""
        X, y, feature_names, target_names = load_dataset()
        assert X.shape[1] == 30  # Breast Cancer Wisconsin has 30 features
        assert len(np.unique(y)) == 2  # Binary classification
        assert len(feature_names) == 30


class TestDataSplitting:
    """Test train/val/test splitting."""
    
    def test_split_proportions(self):
        """Test split maintains correct proportions."""
        X, y, _, _ = load_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test, split_info = split_dataset(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2, track_indices=True
        )
        
        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == X.shape[0]
        assert abs(X_train.shape[0] / X.shape[0] - 0.6) < 0.05
        assert abs(X_val.shape[0] / X.shape[0] - 0.2) < 0.05
        assert abs(X_test.shape[0] / X.shape[0] - 0.2) < 0.05
    
    def test_no_overlap(self):
        """Test no overlapping records between splits."""
        X, y, _, _ = load_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test, split_info = split_dataset(
            X, y, track_indices=True
        )
        
        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0
        
        # Check for index overlap
        train_set = set(split_info['train_indices'])
        val_set = set(split_info['val_indices'])
        test_set = set(split_info['test_indices'])
        
        assert len(train_set & val_set) == 0, "Train and val indices overlap"
        assert len(train_set & test_set) == 0, "Train and test indices overlap"
        assert len(val_set & test_set) == 0, "Val and test indices overlap"
    
    def test_stratification(self):
        """Test stratified split maintains class balance."""
        X, y, _, _ = load_dataset()
        original_ratio = np.mean(y)
        
        X_train, _, _, y_train, _, _, _ = split_dataset(X, y, stratify=True, track_indices=True)
        train_ratio = np.mean(y_train)
        
        # Ratios should be close
        assert abs(original_ratio - train_ratio) < 0.05


class TestPreprocessing:
    """Test preprocessing pipelines."""
    
    def test_classical_preprocessing(self):
        """Test classical preprocessing."""
        X, y, _, _ = load_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test, _ = split_dataset(X, y, track_indices=True)
        
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
        X, y, _, _ = load_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test, _ = split_dataset(X, y, track_indices=True)
        
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


class TestSplitOverlapValidation:
    """Test split overlap checking."""
    
    def test_no_overlap_detected(self):
        """Test that check_split_overlap passes with no overlaps."""
        split_info = {
            'train_indices': np.array([0, 1, 2, 3]),
            'val_indices': np.array([4, 5, 6]),
            'test_indices': np.array([7, 8, 9])
        }
        
        assert check_split_overlap(split_info)
    
    def test_train_val_overlap_detected(self):
        """Test that train/val overlap is detected."""
        split_info = {
            'train_indices': np.array([0, 1, 2, 3]),
            'val_indices': np.array([3, 4, 5, 6]),  # Overlaps at 3
            'test_indices': np.array([7, 8, 9])
        }
        
        with pytest.raises(ValueError, match="Train and validation"):
            check_split_overlap(split_info)
    
    def test_train_test_overlap_detected(self):
        """Test that train/test overlap is detected."""
        split_info = {
            'train_indices': np.array([0, 1, 2, 3]),
            'val_indices': np.array([4, 5, 6]),
            'test_indices': np.array([2, 7, 8, 9])  # Overlaps at 2
        }
        
        with pytest.raises(ValueError, match="Train and test"):
            check_split_overlap(split_info)


class TestDatasetStatistics:
    """Test dataset statistics reporting."""
    
    def test_statistics_reported(self):
        """Test that statistics are correctly computed."""
        X, y, feature_names, target_names = load_dataset()
        
        stats = report_dataset_statistics(X, y, feature_names, target_names, name="Test")
        
        assert stats['n_samples'] == 569
        assert stats['n_features'] == 30
        assert len(stats['class_distribution']) == 2
        assert 'train_indices' not in stats  # Just checking structure


class TestCompleteDataPreparation:
    """Test the complete end-to-end preparation pipeline."""
    
    def test_prepare_data_returns_all_artifacts(self):
        """Test that prepare_data returns all necessary artifacts."""
        data = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
        
        # Check all required keys present
        required_keys = [
            'X_train', 'X_val', 'X_test',
            'X_train_quantum', 'X_val_quantum', 'X_test_quantum',
            'y_train', 'y_val', 'y_test',
            'scaler_classical', 'pca_reducer', 'scaler_quantum',
            'feature_names', 'target_names', 'split_info', 'statistics'
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
    
    def test_prepared_data_dimensions(self):
        """Test that prepared data has correct dimensions."""
        data = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
        
        # Classical
        assert data['X_train'].shape[1] == 30
        assert data['X_val'].shape[1] == 30
        assert data['X_test'].shape[1] == 30
        
        # Quantum
        assert data['X_train_quantum'].shape[1] == 4
        assert data['X_val_quantum'].shape[1] == 4
        assert data['X_test_quantum'].shape[1] == 4
        
        # Labels
        assert len(data['y_train']) == data['X_train'].shape[0]
        assert len(data['y_val']) == data['X_val'].shape[0]
        assert len(data['y_test']) == data['X_test'].shape[0]
    
    def test_split_proportions_correct(self):
        """Test that split proportions are approximately 60/20/20."""
        data = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
        
        n_total = (data['X_train'].shape[0] + 
                   data['X_val'].shape[0] + 
                   data['X_test'].shape[0])
        
        train_ratio = data['X_train'].shape[0] / n_total
        val_ratio = data['X_val'].shape[0] / n_total
        test_ratio = data['X_test'].shape[0] / n_total
        
        assert abs(train_ratio - 0.6) < 0.05
        assert abs(val_ratio - 0.2) < 0.05
        assert abs(test_ratio - 0.2) < 0.05
    
    def test_no_nans_in_prepared_data(self):
        """Test that prepared data contains no NaN or inf values."""
        data = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
        
        assert np.isfinite(data['X_train']).all()
        assert np.isfinite(data['X_val']).all()
        assert np.isfinite(data['X_test']).all()
        assert np.isfinite(data['X_train_quantum']).all()
        assert np.isfinite(data['X_val_quantum']).all()
        assert np.isfinite(data['X_test_quantum']).all()
    
    def test_reproducibility(self):
        """Test that prepare_data produces same results with same seed."""
        data1 = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
        data2 = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
        
        # Classical data should be identical
        np.testing.assert_array_almost_equal(data1['X_train'], data2['X_train'])
        np.testing.assert_array_almost_equal(data1['X_val'], data2['X_val'])
        np.testing.assert_array_almost_equal(data1['X_test'], data2['X_test'])
        
        # Quantum data should be identical
        np.testing.assert_array_almost_equal(data1['X_train_quantum'], data2['X_train_quantum'])
        np.testing.assert_array_almost_equal(data1['X_val_quantum'], data2['X_val_quantum'])
        np.testing.assert_array_almost_equal(data1['X_test_quantum'], data2['X_test_quantum'])
        
        # Labels should be identical
        np.testing.assert_array_equal(data1['y_train'], data2['y_train'])
        np.testing.assert_array_equal(data1['y_val'], data2['y_val'])
        np.testing.assert_array_equal(data1['y_test'], data2['y_test'])



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
