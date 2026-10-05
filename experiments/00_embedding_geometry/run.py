"""Run the deterministic toy embedding geometry comparison."""

import json
import platform
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The directory name begins with a number, so load its local config through
# the script directory rather than as a dotted Python package name.
repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

from src.embeddings import random_embeddings
from src.metrics import pairwise_geometry, ratio_summary, summary
from src.normalization import rms_normalize


def main() -> None:
    """Generate toy matrices and save numerical and graphical artifacts."""
    output_dir = Path(config.OUTPUT_DIR)
    figure_dir = Path(config.FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw = random_embeddings(
        config.VOCABULARY_SIZE,
        config.EMBEDDING_DIM,
        config.SEED,
    )
    rms_normalized = rms_normalize(raw, epsilon=config.NORMALIZATION_EPSILON)
    geometry = pairwise_geometry(raw, rms_normalized)

    np.savez_compressed(
        output_dir / "embedding_matrices.npz",
        raw=raw,
        rms_normalized=rms_normalized,
    )
    np.savez_compressed(output_dir / "pairwise_geometry.npz", **geometry)
    metrics = {
        "config": {
            "seed": config.SEED,
            "vocabulary_size": config.VOCABULARY_SIZE,
            "embedding_dim": config.EMBEDDING_DIM,
            "matrix_shape": list(raw.shape),
            "normalization": {
                "method": config.NORMALIZATION_METHOD,
                "axis": config.NORMALIZATION_AXIS,
                "epsilon": config.NORMALIZATION_EPSILON,
                "formula_convention": config.NORMALIZATION_FORMULA,
            },
            "pair_selection": config.PAIR_SELECTION,
        },
        "reproducibility": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
        },
        "raw": summary(raw),
        "rms_normalized": summary(rms_normalized),
        "pairwise_euclidean_distance_ratio": ratio_summary(
            geometry["distance_ratios"]
        ),
        "pairwise_cosine_similarity": {
            "raw_mean": float(np.mean(geometry["raw_cosines"])),
            "normalized_mean": float(np.mean(geometry["normalized_cosines"])),
            "mean_absolute_change": float(
                np.mean(np.abs(geometry["cosine_differences"]))
            ),
            "max_absolute_change": float(
                np.max(np.abs(geometry["cosine_differences"]))
            ),
            "allclose_atol_1e-12_rtol_1e-12": bool(
                np.allclose(
                    geometry["raw_cosines"],
                    geometry["normalized_cosines"],
                    atol=1e-12,
                    rtol=1e-12,
                )
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(
        geometry["raw_distances"],
        geometry["normalized_distances"],
        s=5,
        alpha=0.25,
    )
    distance_limit = float(
        max(np.max(geometry["raw_distances"]), np.max(geometry["normalized_distances"]))
    )
    axis.plot([0, distance_limit], [0, distance_limit], "k--", linewidth=1)
    axis.set(xlabel="Raw pairwise Euclidean distance", ylabel="RMSNorm distance")
    axis.set_title("Pairwise distance comparison")
    figure.tight_layout()
    figure.savefig(figure_dir / "pairwise_distances.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(
        geometry["raw_cosines"],
        geometry["normalized_cosines"],
        s=5,
        alpha=0.25,
    )
    axis.plot([-1, 1], [-1, 1], "k--", linewidth=1)
    axis.set(
        xlabel="Raw pairwise cosine similarity",
        ylabel="RMSNorm pairwise cosine similarity",
        xlim=(-1, 1),
        ylim=(-1, 1),
    )
    axis.set_title("Pairwise cosine comparison")
    figure.tight_layout()
    figure.savefig(figure_dir / "pairwise_cosines.png", dpi=150)
    plt.close(figure)

    print(
        "Embedding geometry experiment completed: "
        f"seed={config.SEED}, shape={raw.shape}, "
        f"pairs={len(geometry['distance_ratios'])}, "
        f"distance_ratio_mean={metrics['pairwise_euclidean_distance_ratio']['mean']:.6f}, "
        f"cosines_allclose={metrics['pairwise_cosine_similarity']['allclose_atol_1e-12_rtol_1e-12']}"
    )


if __name__ == "__main__":
    main()
