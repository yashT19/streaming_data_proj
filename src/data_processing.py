"""
Data processing: Dataset loading, splitting, and preprocessing.

This module handles:
1. Loading the Breast Cancer Wisconsin dataset from scikit-learn
2. Stratified train/validation/test splitting
3. Classical feature preprocessing (StandardScaler)
4. Quantum feature preprocessing (PCA to 4 features, then scaling)
5. Validation of no data leakage
6. Tracking split indices to verify no overlap
7. Reporting dataset statistics
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_dataset(
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load the Breast Cancer Wisconsin Diagnostic dataset.
    
    Parameters
    ----------
    stratify : bool, optional
        Whether to stratify by class label (default: True)
    random_state : int, optional
        Random state for reproducibility (default: 42)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]
        Features (X), targets (y), feature names, and class names
    """
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    feature_names = np.array(dataset.feature_names)
    target_names = {0: dataset.target_names[0], 1: dataset.target_names[1]}
    
    logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target classes: {target_names}")
    
    return X, y, feature_names, target_names


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    track_indices: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Perform stratified train/validation/test split.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    train_size : float, optional
        Fraction for training (default: 0.6)
    val_size : float, optional
        Fraction for validation (default: 0.2)
    test_size : float, optional
        Fraction for testing (default: 0.2)
    random_state : int, optional
        Random state for reproducibility (default: 42)
    stratify : bool, optional
        Whether to stratify by class (default: True)
    track_indices : bool, optional
        Whether to track original indices for overlap checking (default: True)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]
        X_train, X_val, X_test, y_train, y_val, y_test, and optionally index tracking dict
    """
    n_samples = X.shape[0]
    
    # Create index array for tracking
    indices = np.arange(n_samples) if track_indices else None
    
    # First split: train and temp (val+test)
    stratify_array = y if stratify else None
    if track_indices:
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            X, y, indices,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify_array
        )
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify_array
        )
    
    # Second split: val and test
    val_ratio = val_size / (val_size + test_size)
    if track_indices:
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp,
            train_size=val_ratio,
            random_state=random_state,
            stratify=y_temp if stratify else None
        )
    else:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_ratio,
            random_state=random_state,
            stratify=y_temp if stratify else None
        )
    
    logger.info(f"Split: train {X_train.shape[0]}, val {X_val.shape[0]}, test {X_test.shape[0]}")
    
    # Prepare index tracking dict
    split_info = None
    if track_indices:
        split_info = {
            'train_indices': idx_train,
            'val_indices': idx_val,
            'test_indices': idx_test
        }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, split_info


def preprocess_classical(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Apply classical preprocessing (StandardScaler).
    
    Fit scaler on training data only.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_val : np.ndarray
        Validation features
    X_test : np.ndarray
        Test features
    random_state : int, optional
        Random state (default: 42)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]
        Scaled X_train, X_val, X_test, and fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Classical preprocessing: scaled to mean={X_train_scaled.mean():.3f}, std={X_train_scaled.std():.3f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_quantum(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_features: int = 4,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, StandardScaler]:
    """
    Apply quantum preprocessing (PCA to n_features, then StandardScaler).
    
    Fit PCA and scaler on training data only.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_val : np.ndarray
        Validation features
    X_test : np.ndarray
        Test features
    n_features : int, optional
        Number of features to reduce to (default: 4)
    random_state : int, optional
        Random state for PCA (default: 42)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, StandardScaler]
        Reduced X_train, X_val, X_test, fitted PCA, and fitted scaler
    """
    pca = PCA(n_components=n_features, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    # Scale after PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"Quantum preprocessing: PCA to {n_features} features, "
                f"explained variance={explained_var:.3f}, "
                f"scaled to mean={X_train_scaled.mean():.3f}, std={X_train_scaled.std():.3f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, pca, scaler


def validate_data(
    X: np.ndarray,
    y: np.ndarray,
    name: str = "Dataset"
) -> bool:
    """
    Validate dataset for basic issues.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    name : str, optional
        Name for logging (default: "Dataset")
    
    Returns
    -------
    bool
        True if validation passes
    
    Raises
    ------
    ValueError
        If validation fails
    """
    if X.shape[0] == 0:
        raise ValueError(f"{name}: Empty feature set")
    if y.shape[0] == 0:
        raise ValueError(f"{name}: Empty target set")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{name}: Mismatched lengths: {X.shape[0]} samples vs {y.shape[0]} targets")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: Non-finite values in features")
    if not np.isfinite(y).all():
        raise ValueError(f"{name}: Non-finite values in targets")
    
    logger.info(f"{name} validation passed: {X.shape[0]} samples, {X.shape[1] if X.ndim > 1 else 1} features")
    
    return True


def check_split_overlap(split_info: Dict[str, np.ndarray]) -> bool:
    """
    Check that train, val, and test splits have no overlapping indices.
    
    Parameters
    ----------
    split_info : Dict[str, np.ndarray]
        Dictionary with 'train_indices', 'val_indices', 'test_indices'
    
    Returns
    -------
    bool
        True if no overlaps detected
    
    Raises
    ------
    ValueError
        If overlaps are detected
    """
    train_indices = set(split_info['train_indices'])
    val_indices = set(split_info['val_indices'])
    test_indices = set(split_info['test_indices'])
    
    if train_indices & val_indices:
        raise ValueError("Train and validation sets overlap!")
    if train_indices & test_indices:
        raise ValueError("Train and test sets overlap!")
    if val_indices & test_indices:
        raise ValueError("Validation and test sets overlap!")
    
    logger.info("Split overlap check passed: no overlaps detected")
    return True


def report_dataset_statistics(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[np.ndarray] = None,
    target_names: Optional[Dict[int, str]] = None,
    name: str = "Dataset"
) -> Dict[str, Any]:
    """
    Report comprehensive statistics about the dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    feature_names : np.ndarray, optional
        Feature names
    target_names : Dict[int, str], optional
        Target class names
    name : str, optional
        Name for logging
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing dataset statistics
    """
    stats = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': feature_names,
        'target_names': target_names,
        'unique_classes': np.unique(y),
        'class_distribution': {}
    }
    
    for class_label in np.unique(y):
        count = np.sum(y == class_label)
        pct = 100 * count / len(y)
        class_name = target_names.get(class_label, f"Class {class_label}") if target_names else f"Class {class_label}"
        stats['class_distribution'][class_label] = {
            'count': count,
            'percentage': pct,
            'name': class_name
        }
    
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Samples: {stats['n_samples']}")
    logger.info(f"  Features: {stats['n_features']}")
    logger.info(f"  Classes: {len(stats['unique_classes'])}")
    for class_id, dist in stats['class_distribution'].items():
        logger.info(f"    {dist['name']}: {dist['count']} ({dist['percentage']:.1f}%)")
    
    return stats


def prepare_data(
    random_state: int = 42,
    n_quantum_features: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete end-to-end data preparation pipeline.
    
    This function:
    1. Loads the dataset
    2. Performs stratified train/val/test split
    3. Creates classical and quantum preprocessing
    4. Validates all steps
    5. Reports statistics
    
    Parameters
    ----------
    random_state : int, optional
        Random state for reproducibility (default: 42)
    n_quantum_features : int, optional
        Number of features for quantum preprocessing (default: 4)
    verbose : bool, optional
        Whether to log detailed information (default: True)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - X_train, X_val, X_test (classical)
        - X_train_q, X_val_q, X_test_q (quantum)
        - y_train, y_val, y_test
        - scaler_classical, pca_reducer, scaler_quantum
        - feature_names, target_names
        - split_info (train/val/test indices)
        - statistics (dataset statistics)
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    X, y, feature_names, target_names = load_dataset(random_state=random_state)
    validate_data(X, y, name="Raw dataset")
    
    # Report raw statistics
    raw_stats = report_dataset_statistics(X, y, feature_names, target_names, name="Raw Dataset")
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test, split_info = split_dataset(
        X, y,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=random_state,
        stratify=True,
        track_indices=True
    )
    
    # Verify no overlap
    check_split_overlap(split_info)
    
    # Validate splits
    validate_data(X_train, y_train, name="Training set")
    validate_data(X_val, y_val, name="Validation set")
    validate_data(X_test, y_test, name="Test set")
    
    # Report split statistics
    train_stats = report_dataset_statistics(X_train, y_train, feature_names, target_names, name="Training Set")
    val_stats = report_dataset_statistics(X_val, y_val, feature_names, target_names, name="Validation Set")
    test_stats = report_dataset_statistics(X_test, y_test, feature_names, target_names, name="Test Set")
    
    # Classical preprocessing
    X_train_c, X_val_c, X_test_c, scaler_c = preprocess_classical(
        X_train, X_val, X_test, random_state=random_state
    )
    validate_data(X_train_c, y_train, name="Classical training set")
    validate_data(X_val_c, y_val, name="Classical validation set")
    validate_data(X_test_c, y_test, name="Classical test set")
    
    # Quantum preprocessing
    X_train_q, X_val_q, X_test_q, pca, scaler_q = preprocess_quantum(
        X_train, X_val, X_test, n_features=n_quantum_features, random_state=random_state
    )
    validate_data(X_train_q, y_train, name="Quantum training set")
    validate_data(X_val_q, y_val, name="Quantum validation set")
    validate_data(X_test_q, y_test, name="Quantum test set")
    
    # Verify quantum reduced to correct dimensions
    assert X_train_q.shape[1] == n_quantum_features, f"Quantum features should be {n_quantum_features}, got {X_train_q.shape[1]}"
    assert X_val_q.shape[1] == n_quantum_features
    assert X_test_q.shape[1] == n_quantum_features
    
    logger.info(f"\n✓ Data preparation complete!")
    logger.info(f"  Classical: {X_train_c.shape[0]} train × {X_train_c.shape[1]} features")
    logger.info(f"  Quantum: {X_train_q.shape[0]} train × {X_train_q.shape[1]} features")
    
    return {
        # Classical preprocessing
        'X_train': X_train_c,
        'X_val': X_val_c,
        'X_test': X_test_c,
        # Quantum preprocessing
        'X_train_quantum': X_train_q,
        'X_val_quantum': X_val_q,
        'X_test_quantum': X_test_q,
        # Labels
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        # Preprocessing objects
        'scaler_classical': scaler_c,
        'pca_reducer': pca,
        'scaler_quantum': scaler_q,
        # Metadata
        'feature_names': feature_names,
        'target_names': target_names,
        'split_info': split_info,
        # Statistics
        'statistics': {
            'raw': raw_stats,
            'train': train_stats,
            'val': val_stats,
            'test': test_stats
        }
    }


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    data = prepare_data(random_state=42, n_quantum_features=4, verbose=True)
    
    print("\n" + "="*60)
    print("Data Preparation Summary")
    print("="*60)
    print(f"Classical training shape: {data['X_train'].shape}")
    print(f"Quantum training shape: {data['X_train_quantum'].shape}")
    print(f"Validation shape: {data['X_val'].shape}")
    print(f"Test shape: {data['X_test'].shape}")
    print(f"Train labels distribution: {np.bincount(data['y_train'])}")
    print(f"Test labels distribution: {np.bincount(data['y_test'])}")
    print("✓ Data processing smoke test passed.")
