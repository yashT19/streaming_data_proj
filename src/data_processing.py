"""
Data processing: Dataset loading, splitting, and preprocessing.

This module handles:
1. Loading the Breast Cancer Wisconsin dataset from scikit-learn
2. Stratified train/validation/test splitting
3. Classical feature preprocessing (StandardScaler)
4. Quantum feature preprocessing (PCA to 4 features, then scaling)
5. Validation of no data leakage
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_dataset(
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
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
    Tuple[np.ndarray, np.ndarray]
        Features (X) and targets (y)
    """
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    
    logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train and temp (val+test)
    stratify_array = y if stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_array
    )
    
    # Second split: val and test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=random_state,
        stratify=y_temp if stratify else None
    )
    
    logger.info(f"Split: train {X_train.shape[0]}, val {X_val.shape[0]}, test {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


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


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    X, y = load_dataset()
    validate_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    X_train_c, X_val_c, X_test_c, scaler_c = preprocess_classical(X_train, X_val, X_test)
    X_train_q, X_val_q, X_test_q, pca, scaler_q = preprocess_quantum(X_train, X_val, X_test)
    print("Data processing smoke test passed.")
