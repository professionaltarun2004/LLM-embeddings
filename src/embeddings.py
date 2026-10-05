"""Creation of toy embedding matrices."""

import numpy as np


def random_embeddings(
    vocabulary_size: int,
    embedding_dim: int,
    seed: int,
) -> np.ndarray:
    """Create a reproducible standard-normal embedding matrix."""
    if vocabulary_size <= 0 or embedding_dim <= 0:
        raise ValueError("vocabulary_size and embedding_dim must be positive")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((vocabulary_size, embedding_dim))
