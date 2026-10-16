#!/usr/bin/env python
"""
Comprehensive verification script for the data preprocessing pipeline.

This script:
1. Loads and validates the dataset
2. Performs stratified splitting with overlap checking
3. Applies preprocessing (classical and quantum)
4. Verifies all key properties
5. Reports detailed statistics
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import prepare_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def verify_pipeline():
    """Run complete pipeline verification."""
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING PIPELINE VERIFICATION")
    print("="*80 + "\n")
    
    # Prepare data
    print("Step 1: Loading and preparing data...")
    data = prepare_data(random_state=42, n_quantum_features=4, verbose=True)
    
    print("\n" + "-"*80)
    print("Step 2: Verifying shapes and dimensions...")
    print("-"*80)
    
    # Dataset shape checks
    checks = {
        "Classical training shape": (data['X_train'].shape, (341, 30)),
        "Classical validation shape": (data['X_val'].shape, (114, 30)),
        "Classical test shape": (data['X_test'].shape, (114, 30)),
        "Quantum training shape": (data['X_train_quantum'].shape, (341, 4)),
        "Quantum validation shape": (data['X_val_quantum'].shape, (114, 4)),
        "Quantum test shape": (data['X_test_quantum'].shape, (114, 4)),
        "Train labels shape": (data['y_train'].shape, (341,)),
        "Validation labels shape": (data['y_val'].shape, (114,)),
        "Test labels shape": (data['y_test'].shape, (114,)),
    }
    
    all_passed = True
    for check_name, (actual, expected) in checks.items():
        passed = actual == expected
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
        print(f"       Expected: {expected}, Got: {actual}")
        if not passed:
            all_passed = False
    
    print("\n" + "-"*80)
    print("Step 3: Verifying feature scaling and NaN/Inf values...")
    print("-"*80)
    
    # Classical scaling check
    classical_mean = np.mean(data['X_train'])
    classical_std = np.std(data['X_train'])
    classical_check = abs(classical_mean) < 0.01 and abs(classical_std - 1.0) < 0.01
    print(f"{'✓ PASS' if classical_check else '✗ FAIL'}: Classical features scaled")
    print(f"       Mean ≈ 0: {classical_mean:.6f}, Std ≈ 1: {classical_std:.6f}")
    
    # Quantum scaling check
    quantum_mean = np.mean(data['X_train_quantum'])
    quantum_std = np.std(data['X_train_quantum'])
    quantum_check = abs(quantum_mean) < 0.01 and abs(quantum_std - 1.0) < 0.01
    print(f"{'✓ PASS' if quantum_check else '✗ FAIL'}: Quantum features scaled")
    print(f"       Mean ≈ 0: {quantum_mean:.6f}, Std ≈ 1: {quantum_std:.6f}")
    
    # NaN/Inf checks
    has_nans = (
        np.isnan(data['X_train']).any() or np.isnan(data['X_val']).any() or np.isnan(data['X_test']).any() or
        np.isnan(data['X_train_quantum']).any() or np.isnan(data['X_val_quantum']).any() or np.isnan(data['X_test_quantum']).any()
    )
    has_infs = (
        np.isinf(data['X_train']).any() or np.isinf(data['X_val']).any() or np.isinf(data['X_test']).any() or
        np.isinf(data['X_train_quantum']).any() or np.isinf(data['X_val_quantum']).any() or np.isinf(data['X_test_quantum']).any()
    )
    
    nan_check = not has_nans
    inf_check = not has_infs
    
    print(f"{'✓ PASS' if nan_check else '✗ FAIL'}: No NaN values found")
    print(f"{'✓ PASS' if inf_check else '✗ FAIL'}: No Inf values found")
    
    print("\n" + "-"*80)
    print("Step 4: Verifying split proportions...")
    print("-"*80)
    
    n_total = data['X_train'].shape[0] + data['X_val'].shape[0] + data['X_test'].shape[0]
    train_pct = 100 * data['X_train'].shape[0] / n_total
    val_pct = 100 * data['X_val'].shape[0] / n_total
    test_pct = 100 * data['X_test'].shape[0] / n_total
    
    proportion_ok = (
        abs(train_pct - 60) < 5 and 
        abs(val_pct - 20) < 5 and 
        abs(test_pct - 20) < 5
    )
    
    print(f"{'✓ PASS' if proportion_ok else '✗ FAIL'}: Split proportions approximately 60/20/20")
    print(f"       Train: {train_pct:.1f}% ({data['X_train'].shape[0]} records)")
    print(f"       Val:   {val_pct:.1f}% ({data['X_val'].shape[0]} records)")
    print(f"       Test:  {test_pct:.1f}% ({data['X_test'].shape[0]} records)")
    
    print("\n" + "-"*80)
    print("Step 5: Verifying no overlapping indices between splits...")
    print("-"*80)
    
    split_info = data['split_info']
    train_set = set(split_info['train_indices'])
    val_set = set(split_info['val_indices'])
    test_set = set(split_info['test_indices'])
    
    train_val_overlap = len(train_set & val_set)
    train_test_overlap = len(train_set & test_set)
    val_test_overlap = len(val_set & test_set)
    
    overlap_check = (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0)
    
    print(f"{'✓ PASS' if overlap_check else '✗ FAIL'}: No overlapping indices")
    print(f"       Train ∩ Val:  {train_val_overlap} overlaps")
    print(f"       Train ∩ Test: {train_test_overlap} overlaps")
    print(f"       Val ∩ Test:   {val_test_overlap} overlaps")
    
    print("\n" + "-"*80)
    print("Step 6: Verifying label distributions...")
    print("-"*80)
    
    for split_name, y_split in [('Train', data['y_train']), 
                                 ('Validation', data['y_val']), 
                                 ('Test', data['y_test'])]:
        unique, counts = np.unique(y_split, return_counts=True)
        pcts = 100 * counts / len(y_split)
        print(f"{split_name}:")
        print(f"       Malignant (0): {counts[0]:3d} ({pcts[0]:.1f}%)")
        print(f"       Benign (1):    {counts[1]:3d} ({pcts[1]:.1f}%)")
    
    print("\n" + "-"*80)
    print("Step 7: Verifying preprocessing objects...")
    print("-"*80)
    
    scaler_c = data['scaler_classical']
    pca = data['pca_reducer']
    scaler_q = data['scaler_quantum']
    
    print(f"✓ PASS: Classical scaler fitted")
    print(f"       Mean: {scaler_c.mean_[:3]}... (30 features)")
    print(f"       Scale: {scaler_c.scale_[:3]}... (30 features)")
    
    print(f"✓ PASS: PCA reducer fitted to {pca.n_components_} components")
    print(f"       Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    print(f"✓ PASS: Quantum scaler fitted")
    print(f"       Mean: {scaler_q.mean_[:2]}... (4 features)")
    print(f"       Scale: {scaler_q.scale_[:2]}... (4 features)")
    
    print("\n" + "-"*80)
    print("Step 8: Verifying reproducibility (repeated runs with same seed)...")
    print("-"*80)
    
    data_rerun = prepare_data(random_state=42, n_quantum_features=4, verbose=False)
    
    classical_matches = np.allclose(data['X_train'], data_rerun['X_train'])
    quantum_matches = np.allclose(data['X_train_quantum'], data_rerun['X_train_quantum'])
    labels_match = np.array_equal(data['y_train'], data_rerun['y_train'])
    
    reproducibility_check = classical_matches and quantum_matches and labels_match
    
    print(f"{'✓ PASS' if classical_matches else '✗ FAIL'}: Classical preprocessing reproducible")
    print(f"{'✓ PASS' if quantum_matches else '✗ FAIL'}: Quantum preprocessing reproducible")
    print(f"{'✓ PASS' if labels_match else '✗ FAIL'}: Labels reproducible")
    
    print("\n" + "-"*80)
    print("Step 9: Verifying metadata...")
    print("-"*80)
    
    feature_names = data['feature_names']
    target_names = data['target_names']
    
    print(f"✓ PASS: Feature names available")
    print(f"       Total features: {len(feature_names)}")
    print(f"       First 3: {feature_names[:3]}")
    
    print(f"✓ PASS: Target names available")
    print(f"       0 -> {target_names[0]}")
    print(f"       1 -> {target_names[1]}")
    
    print("\n" + "="*80)
    print("PIPELINE VERIFICATION SUMMARY")
    print("="*80)
    
    all_checks = [
        all_passed,
        classical_check and quantum_check,
        nan_check and inf_check,
        proportion_ok,
        overlap_check,
        reproducibility_check
    ]
    
    if all(all_checks):
        print("✓ ALL CHECKS PASSED")
        print("\nThe data preprocessing pipeline is working correctly!")
        print("Ready for model training and experimentation.")
        return True
    else:
        print("✗ SOME CHECKS FAILED")
        print("Please review the output above for details.")
        return False


if __name__ == "__main__":
    success = verify_pipeline()
    sys.exit(0 if success else 1)
