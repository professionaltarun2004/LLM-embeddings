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


def pairwise_geometry(
    raw: np.ndarray,
    normalized: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compare distances and cosine similarities for every unordered row pair.

    This exact O(V^2) calculation is intended for the small toy vocabulary.
    Each returned observation uses the same original row-index pair.
    """
    if raw.shape != normalized.shape or raw.ndim != 2:
        raise ValueError("raw and normalized must be matching 2D matrices")

    row_i, row_j = np.triu_indices(raw.shape[0], k=1)
    raw_i, raw_j = raw[row_i], raw[row_j]
    normalized_i, normalized_j = normalized[row_i], normalized[row_j]

    raw_distances = np.linalg.norm(raw_i - raw_j, axis=1)
    normalized_distances = np.linalg.norm(normalized_i - normalized_j, axis=1)
    distance_ratios = normalized_distances / np.maximum(raw_distances, 1e-8)

    raw_cosines = np.sum(raw_i * raw_j, axis=1) / (
        np.linalg.norm(raw_i, axis=1) * np.linalg.norm(raw_j, axis=1)
    )
    normalized_cosines = np.sum(normalized_i * normalized_j, axis=1) / (
        np.linalg.norm(normalized_i, axis=1) * np.linalg.norm(normalized_j, axis=1)
    )

    return {
        "row_i": row_i,
        "row_j": row_j,
        "raw_distances": raw_distances,
        "normalized_distances": normalized_distances,
        "distance_ratios": distance_ratios,
        "raw_cosines": raw_cosines,
        "normalized_cosines": normalized_cosines,
        "cosine_differences": normalized_cosines - raw_cosines,
    }


def ratio_summary(values: np.ndarray) -> dict[str, float | dict[str, float]]:
    """Summarize pairwise distance ratios and selected percentiles."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {
            str(percentile): float(np.percentile(values, percentile))
            for percentile in (5, 25, 75, 95)
        },
    }
