"""Basic descriptive measurements for embedding matrices."""

import numpy as np


def row_norms(values: np.ndarray) -> np.ndarray:
    """Return the Euclidean norm of each embedding row."""
    return np.linalg.norm(values, axis=-1)


def summary(values: np.ndarray) -> dict[str, float]:
    """Return compact aggregate statistics for an embedding matrix."""
    norms = row_norms(values)
    return {
        "mean_row_norm": float(np.mean(norms)),
        "std_row_norm": float(np.std(norms)),
        "mean_coordinate": float(np.mean(values)),
        "std_coordinate": float(np.std(values)),
    }
