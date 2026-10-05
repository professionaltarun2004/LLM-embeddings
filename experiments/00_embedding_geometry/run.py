"""Run the minimal, deterministic embedding geometry scaffold."""

import json
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
from src.metrics import row_norms, summary
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
    rms_normalized = rms_normalize(raw)

    np.savez_compressed(
        output_dir / "embedding_matrices.npz",
        raw=raw,
        rms_normalized=rms_normalized,
    )
    metrics = {
        "config": {
            "seed": config.SEED,
            "vocabulary_size": config.VOCABULARY_SIZE,
            "embedding_dim": config.EMBEDDING_DIM,
        },
        "raw": summary(raw),
        "rms_normalized": summary(rms_normalized),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(6, 4))
    raw_norms = row_norms(raw)
    normalized_norms = row_norms(rms_normalized)
    bins = np.linspace(0.0, float(np.max(raw_norms)) * 1.05, 21)
    plt.hist(raw_norms, bins=bins, alpha=0.65, label="raw")
    plt.hist(normalized_norms, bins=bins, alpha=0.65, label="RMS-normalized")
    plt.xlabel("Embedding row L2 norm")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "row_norms.png", dpi=150)
    plt.close()

    print(
        "Embedding geometry scaffold completed: "
        f"seed={config.SEED}, shape={raw.shape}, "
        f"numeric_results={output_dir}, figure={figure_dir / 'row_norms.png'}"
    )


if __name__ == "__main__":
    main()
