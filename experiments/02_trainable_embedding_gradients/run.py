"""Compare embedding-table gradients for two RMSNorm placements."""

import json
import platform
from pathlib import Path
import sys

import numpy as np
import torch

repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

from src.embeddings import random_embeddings


def rms_normalize(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Apply the project's RMSNorm formula along the last (feature) axis."""
    rms = torch.sqrt(torch.mean(torch.square(values), dim=-1, keepdim=True))
    return values / torch.clamp_min(rms, epsilon)


def main() -> None:
    """Measure forward, loss, and E-gradient differences without updating E."""
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_values = random_embeddings(
        config.VOCABULARY_SIZE,
        config.EMBEDDING_DIM,
        config.SEED,
    )
    embedding_table = torch.tensor(initial_values, dtype=torch.float64, requires_grad=True)

    # Both conceptual paths reference the same trainable tensor E.
    embedding_table_a = embedding_table
    embedding_table_b = embedding_table
    initial_values_a = embedding_table_a.detach().cpu().numpy()
    initial_values_b = embedding_table_b.detach().cpu().numpy()
    same_initial_matrix_object = embedding_table_a is embedding_table_b
    same_initial_matrix_values = np.array_equal(initial_values_a, initial_values_b)
    assert same_initial_matrix_object and same_initial_matrix_values

    token_ids = torch.tensor(config.TOKEN_IDS, dtype=torch.long)
    if torch.any(token_ids < 0) or torch.any(token_ids >= config.VOCABULARY_SIZE):
        raise ValueError("TOKEN_IDS must contain valid embedding-table row indices")

    # One fixed target is shared by both conditions. It is deterministic and
    # contains nonzero values, so the mean-squared-error loss has a useful
    # gradient with respect to the selected embedding rows.
    target_values = np.linspace(
        config.TARGET_MIN,
        config.TARGET_MAX,
        num=len(config.TOKEN_IDS) * config.EMBEDDING_DIM,
        dtype=np.float64,
    ).reshape(len(config.TOKEN_IDS), config.EMBEDDING_DIM)
    target = torch.tensor(target_values, dtype=torch.float64)

    # Path A: E[token_ids] followed by row-wise RMSNorm.
    representation_a = rms_normalize(
        embedding_table_a[token_ids],
        epsilon=config.NORMALIZATION_EPSILON,
    )

    # Path B: row-wise RMSNorm(E) followed by lookup from that same tensor.
    normalized_table_b = rms_normalize(
        embedding_table_b,
        epsilon=config.NORMALIZATION_EPSILON,
    )
    representation_b = normalized_table_b[token_ids]

    loss_a = torch.mean(torch.square(representation_a - target))
    loss_b = torch.mean(torch.square(representation_b - target))

    gradient_a = torch.autograd.grad(loss_a, embedding_table, retain_graph=True)[0]
    gradient_b = torch.autograd.grad(loss_b, embedding_table)[0]

    representation_a_np = representation_a.detach().cpu().numpy()
    representation_b_np = representation_b.detach().cpu().numpy()
    gradient_a_np = gradient_a.detach().cpu().numpy()
    gradient_b_np = gradient_b.detach().cpu().numpy()
    representation_absolute_difference = np.abs(representation_a_np - representation_b_np)
    gradient_absolute_difference = np.abs(gradient_a_np - gradient_b_np)

    representation_reference = np.abs(representation_a_np)
    representation_relative_mask = (
        representation_reference > config.RELATIVE_DIFFERENCE_DENOMINATOR_FLOOR
    )
    representation_relative = (
        representation_absolute_difference[representation_relative_mask]
        / representation_reference[representation_relative_mask]
    )
    gradient_reference = np.abs(gradient_a_np)
    gradient_relative_mask = gradient_reference > config.RELATIVE_DIFFERENCE_DENOMINATOR_FLOOR
    gradient_relative = (
        gradient_absolute_difference[gradient_relative_mask]
        / gradient_reference[gradient_relative_mask]
    )

    loss_a_value = float(loss_a.detach().cpu().item())
    loss_b_value = float(loss_b.detach().cpu().item())
    gradient_difference = gradient_a_np - gradient_b_np
    forward_equal = bool(
        np.allclose(
            representation_a_np,
            representation_b_np,
            atol=config.EQUALITY_ATOL,
            rtol=config.EQUALITY_RTOL,
        )
    )
    loss_equal = bool(
        np.isclose(
            loss_a_value,
            loss_b_value,
            atol=config.EQUALITY_ATOL,
            rtol=config.EQUALITY_RTOL,
        )
    )
    gradients_equal = bool(
        np.allclose(
            gradient_a_np,
            gradient_b_np,
            atol=config.EQUALITY_ATOL,
            rtol=config.EQUALITY_RTOL,
        )
    )

    metrics = {
        "config": {
            "seed": config.SEED,
            "vocabulary_size": config.VOCABULARY_SIZE,
            "embedding_dimension": config.EMBEDDING_DIM,
            "matrix_shape": list(initial_values.shape),
            "token_ids": list(config.TOKEN_IDS),
            "normalization": {
                "method": config.NORMALIZATION_METHOD,
                "axis": config.NORMALIZATION_AXIS,
                "epsilon": config.NORMALIZATION_EPSILON,
                "formula_convention": config.NORMALIZATION_FORMULA,
            },
            "downstream_computation": config.DOWNSTREAM_COMPUTATION,
            "target": {
                "construction": "float64 linspace from target_min to target_max, reshaped to token_count x embedding_dimension",
                "target_min": config.TARGET_MIN,
                "target_max": config.TARGET_MAX,
                "shared_by_both_conditions": True,
            },
            "loss_definition": config.LOSS_DEFINITION,
            "equality_tolerance": {
                "atol": config.EQUALITY_ATOL,
                "rtol": config.EQUALITY_RTOL,
            },
            "relative_difference_denominator_floor": config.RELATIVE_DIFFERENCE_DENOMINATOR_FLOOR,
            "optimizer_used": False,
            "parameter_updates_performed": False,
        },
        "reproducibility": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "autodiff_framework": "PyTorch",
            "autodiff_framework_version": torch.__version__,
        },
        "initial_embedding_matrix": {
            "same_tensor_object_for_both_paths": bool(same_initial_matrix_object),
            "exactly_equal_values": bool(same_initial_matrix_values),
        },
        "forward_representation": {
            "maximum_absolute_difference": float(np.max(representation_absolute_difference)),
            "mean_absolute_difference": float(np.mean(representation_absolute_difference)),
            "maximum_relative_difference_where_reference_exceeds_floor": float(
                np.max(representation_relative)
            ),
            "relative_elements_included": int(representation_relative.size),
            "numerically_equal_within_tolerance": forward_equal,
        },
        "loss": {
            "path_a": loss_a_value,
            "path_b": loss_b_value,
            "absolute_difference": abs(loss_a_value - loss_b_value),
            "numerically_equal_within_tolerance": loss_equal,
        },
        "embedding_table_gradient": {
            "maximum_absolute_difference": float(np.max(gradient_absolute_difference)),
            "mean_absolute_difference": float(np.mean(gradient_absolute_difference)),
            "maximum_relative_difference_where_reference_exceeds_floor": float(
                np.max(gradient_relative)
            ),
            "relative_elements_included": int(gradient_relative.size),
            "gradient_a_l2_norm": float(np.linalg.norm(gradient_a_np)),
            "gradient_b_l2_norm": float(np.linalg.norm(gradient_b_np)),
            "gradient_difference_l2_norm": float(np.linalg.norm(gradient_difference)),
            "numerically_equal_within_tolerance": gradients_equal,
        },
    }

    np.savez_compressed(
        output_dir / "gradients.npz",
        initial_embedding_table=initial_values,
        token_ids=np.asarray(config.TOKEN_IDS, dtype=np.int64),
        target=target_values,
        representation_a=representation_a_np,
        representation_b=representation_b_np,
        representation_absolute_difference=representation_absolute_difference,
        gradient_a=gradient_a_np,
        gradient_b=gradient_b_np,
        gradient_difference=gradient_difference,
        gradient_absolute_difference=gradient_absolute_difference,
        loss_a=np.asarray(loss_a_value),
        loss_b=np.asarray(loss_b_value),
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Trainable embedding gradient comparison completed: "
        f"tokens={list(config.TOKEN_IDS)}, "
        f"forward_equal={forward_equal}, loss_equal={loss_equal}, "
        f"gradients_equal={gradients_equal}, "
        f"gradient_difference_l2={metrics['embedding_table_gradient']['gradient_difference_l2_norm']:.3e}"
    )


if __name__ == "__main__":
    main()
