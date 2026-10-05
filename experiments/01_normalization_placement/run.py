"""Compare two RMSNorm placement paths with a fixed embedding table."""

import json
import platform
from pathlib import Path
import sys

import numpy as np

# The experiment directory begins with a number, so load its local config
# directly while keeping imports of project utilities rooted at the repo.
repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

from src.embeddings import random_embeddings
from src.normalization import rms_normalize


def main() -> None:
    """Run both pathways and save their fixed-table comparison."""
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_table = random_embeddings(
        config.VOCABULARY_SIZE,
        config.EMBEDDING_DIM,
        config.SEED,
    )

    # Both paths receive the very same array, not independent copies or draws.
    raw_table_for_post_lookup = raw_table
    raw_table_for_pre_normalization = raw_table
    same_raw_table_object = raw_table_for_post_lookup is raw_table_for_pre_normalization
    same_raw_table_values = np.array_equal(
        raw_table_for_post_lookup,
        raw_table_for_pre_normalization,
    )
    assert same_raw_table_object and same_raw_table_values

    token_ids = np.asarray(config.TOKEN_IDS, dtype=np.int64)
    if token_ids.ndim != 1 or np.any(token_ids < 0) or np.any(
        token_ids >= config.VOCABULARY_SIZE
    ):
        raise ValueError("TOKEN_IDS must be a one-dimensional list of valid row indices")

    # A: lookup from the raw table, then normalize each looked-up vector.
    post_lookup_normalization = rms_normalize(
        raw_table_for_post_lookup[token_ids],
        epsilon=config.NORMALIZATION_EPSILON,
    )

    # B: normalize every table row, then look up those same token IDs.
    normalized_table = rms_normalize(
        raw_table_for_pre_normalization,
        epsilon=config.NORMALIZATION_EPSILON,
    )
    pre_normalized_table = normalized_table[token_ids]

    elementwise_absolute_difference = np.abs(
        post_lookup_normalization - pre_normalized_table
    )
    reference_magnitudes = np.abs(post_lookup_normalization)
    meaningful_relative = reference_magnitudes > config.RELATIVE_DIFFERENCE_DENOMINATOR_FLOOR
    relative_differences = (
        elementwise_absolute_difference[meaningful_relative]
        / reference_magnitudes[meaningful_relative]
    )

    per_token_euclidean_difference = np.linalg.norm(
        post_lookup_normalization - pre_normalized_table,
        axis=-1,
    )
    row_i, row_j = np.triu_indices(len(token_ids), k=1)
    post_i = post_lookup_normalization[row_i]
    post_j = post_lookup_normalization[row_j]
    pre_i = pre_normalized_table[row_i]
    pre_j = pre_normalized_table[row_j]
    post_pairwise_distances = np.linalg.norm(post_i - post_j, axis=1)
    pre_pairwise_distances = np.linalg.norm(pre_i - pre_j, axis=1)
    pairwise_euclidean_distance_differences = np.abs(
        post_pairwise_distances - pre_pairwise_distances
    )

    post_pairwise_cosines = np.sum(post_i * post_j, axis=1) / (
        np.linalg.norm(post_i, axis=1) * np.linalg.norm(post_j, axis=1)
    )
    pre_pairwise_cosines = np.sum(pre_i * pre_j, axis=1) / (
        np.linalg.norm(pre_i, axis=1) * np.linalg.norm(pre_j, axis=1)
    )
    pairwise_cosine_differences = np.abs(
        post_pairwise_cosines - pre_pairwise_cosines
    )

    numerically_equal = bool(
        np.allclose(
            post_lookup_normalization,
            pre_normalized_table,
            atol=config.EQUALITY_ATOL,
            rtol=config.EQUALITY_RTOL,
        )
    )
    metrics = {
        "config": {
            "seed": config.SEED,
            "vocabulary_size": config.VOCABULARY_SIZE,
            "embedding_dimension": config.EMBEDDING_DIM,
            "matrix_shape": list(raw_table.shape),
            "token_ids": token_ids.tolist(),
            "normalization": {
                "method": config.NORMALIZATION_METHOD,
                "axis": config.NORMALIZATION_AXIS,
                "epsilon": config.NORMALIZATION_EPSILON,
                "formula_convention": config.NORMALIZATION_FORMULA,
            },
            "equality_tolerance": {
                "atol": config.EQUALITY_ATOL,
                "rtol": config.EQUALITY_RTOL,
            },
            "relative_difference_denominator_floor": config.RELATIVE_DIFFERENCE_DENOMINATOR_FLOOR,
        },
        "reproducibility": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "same_raw_embedding_matrix": {
            "same_object": bool(same_raw_table_object),
            "exactly_equal_values": bool(same_raw_table_values),
        },
        "elementwise_difference": {
            "maximum_absolute": float(np.max(elementwise_absolute_difference)),
            "mean_absolute": float(np.mean(elementwise_absolute_difference)),
            "maximum_relative_where_reference_magnitude_exceeds_floor": float(
                np.max(relative_differences)
            ),
            "relative_elements_included": int(relative_differences.size),
        },
        "representation_difference": {
            "per_token_euclidean_difference_maximum": float(
                np.max(per_token_euclidean_difference)
            ),
            "per_token_euclidean_difference_mean": float(
                np.mean(per_token_euclidean_difference)
            ),
            "pairwise_euclidean_distance_difference_maximum": float(
                np.max(pairwise_euclidean_distance_differences)
            ),
            "pairwise_euclidean_distance_difference_mean": float(
                np.mean(pairwise_euclidean_distance_differences)
            ),
            "pairwise_cosine_similarity_difference_maximum": float(
                np.max(pairwise_cosine_differences)
            ),
            "pairwise_cosine_similarity_difference_mean": float(
                np.mean(pairwise_cosine_differences)
            ),
        },
        "fixed_table_forward": {
            "numerically_equal_within_tolerance": numerically_equal,
            "trainable_parameter_behavior_evaluated": False,
        },
    }

    np.savez_compressed(
        output_dir / "representations.npz",
        raw_embedding_table=raw_table,
        token_ids=token_ids,
        post_lookup_normalization=post_lookup_normalization,
        normalized_embedding_table=normalized_table,
        pre_normalized_table_lookup=pre_normalized_table,
        elementwise_absolute_difference=elementwise_absolute_difference,
        per_token_euclidean_difference=per_token_euclidean_difference,
        pair_row_i=row_i,
        pair_row_j=row_j,
        post_pairwise_euclidean_distances=post_pairwise_distances,
        pre_pairwise_euclidean_distances=pre_pairwise_distances,
        pairwise_euclidean_distance_differences=pairwise_euclidean_distance_differences,
        post_pairwise_cosines=post_pairwise_cosines,
        pre_pairwise_cosines=pre_pairwise_cosines,
        pairwise_cosine_differences=pairwise_cosine_differences,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Normalization placement comparison completed: "
        f"tokens={len(token_ids)}, shape={post_lookup_normalization.shape}, "
        f"max_abs_difference={metrics['elementwise_difference']['maximum_absolute']:.3e}, "
        f"numerically_equal={numerically_equal}"
    )


if __name__ == "__main__":
    main()
