"""Normalization operations for embedding matrices."""

import numpy as np


def rms_normalize(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Scale each row to unit root-mean-square magnitude."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rms = np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True))
    return values / np.maximum(rms, epsilon)
