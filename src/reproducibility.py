"""
Reproducibility utilities: seeding, random state management, and validation.
"""

import numpy as np
import random
import os


def set_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value (default: 42)
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed_state() -> dict:
    """
    Get current state of random seeds for logging.
    
    Returns
    -------
    dict
        Dictionary containing seed state information
    """
    return {
        'numpy_state': np.random.get_state()[0],
        'random_state': random.getstate()[0],
        'hash_seed': os.environ.get('PYTHONHASHSEED', 'not set'),
    }


def validate_reproducibility(func, seed: int = 42, repetitions: int = 3) -> bool:
    """
    Validate that a function produces reproducible results.
    
    Parameters
    ----------
    func : callable
        Function to validate (must return array-like)
    seed : int, optional
        Seed value (default: 42)
    repetitions : int, optional
        Number of repetitions to compare (default: 3)
    
    Returns
    -------
    bool
        True if all repetitions produce identical results
    """
    results = []
    for _ in range(repetitions):
        set_seeds(seed)
        result = func()
        results.append(result)
    
    # Compare all results
    for i in range(1, len(results)):
        if isinstance(results[i], np.ndarray):
            if not np.allclose(results[i], results[0], rtol=1e-5, atol=1e-8):
                return False
        else:
            if results[i] != results[0]:
                return False
    
    return True


if __name__ == "__main__":
    set_seeds(42)
    print("Seeds set for reproducibility.")
