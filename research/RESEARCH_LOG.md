# Research Log: Embedding Normalization Placement

This file is the chronological source of truth for the investigation. Results and interpretations below are limited to the experiments actually run.

## Research question

> What happens to the information and geometry of token embeddings when normalization is incorporated into the embedding representation itself, rather than applied after embedding lookup—and does the placement of normalization affect downstream model behavior?

## Current status

Experiments 0 and 1 are complete. Experiment 0 measured row and pairwise geometry changes on a seeded synthetic embedding matrix. Experiment 1 compared two RMSNorm placements for a fixed table and found numerically equal outputs under the tested convention and tolerance. No training, gradient, optimization, learned embedding, or Transformer experiment has been run. Experiment 2 is the planned next investigation, not a completed result.

## Investigation flow

```mermaid
flowchart TD
    Q[Original research question]
    E0[Experiment 0: synthetic embedding geometry]
    G[Cosine similarities preserved to numerical precision;<br/>Euclidean distances changed]
    E1[Experiment 1: fixed-table normalization placement]
    F[Fixed-table forward outputs numerically equal<br/>under tested RMSNorm convention and tolerance]
    U[Question shifts from forward computation<br/>to trainable-parameter behavior]
    E2[Experiment 2: trainable behavior<br/>(planned; not implemented)]
    T[Measure gradients, optimization dynamics,<br/>parameter evolution, and learned representations]
    D[Decide whether a larger-scale model experiment<br/>is justified by those results]

    Q --> E0 --> G --> E1 --> F --> U --> E2 --> T --> D
```

## Experiment checkpoint table

| Checkpoint | Question | Status and observed result | Commit |
|---|---|---|---|
| [Experiment 0 — embedding geometry](../experiments/00_embedding_geometry/) | What does row-wise RMSNorm do to a toy embedding matrix's magnitudes and geometry? | Complete. Row magnitudes changed, cosine similarities were preserved to numerical precision, and pairwise Euclidean distances changed. | `f3c34aaf5953e66b5980f1615cd7b81f08b46195` — `experiment: measure embedding geometry under RMSNorm` |
| [Experiment 1 — normalization placement](../experiments/01_normalization_placement/) | With a fixed embedding table, do post-lookup RMSNorm and a pre-normalized table produce different outputs? | Complete. The two paths produced equal outputs under `atol=1e-12`, `rtol=1e-12`. | `d8f88daa35ca89060a7890d2ddbc910518c4f3f4` — `experiment: test normalization placement equivalence` |
| Experiment 2 — trainable behavior | Does normalization parameterization affect gradients, optimization, parameter evolution, or learned representations? | Planned; not implemented or run. | — |

## Experiment 0 checkpoint — embedding geometry

**Purpose.** Measure row-wise RMSNorm effects on a controlled synthetic embedding matrix before comparing normalization placement.

**Setup.** A standard-normal matrix with vocabulary size 128 and embedding dimension 32 was generated with seed 1729. The same rows were used before and after RMSNorm with axis `-1` and epsilon `1e-8`; the convention was `x / max(sqrt(mean(x**2, axis=-1, keepdims=True)), epsilon)`. All 8,128 unique row pairs were measured.

**Observed measurements.** The raw mean row L2 norm was `5.509095` (standard deviation `0.686539`); after RMSNorm the mean was `5.656854` (standard deviation approximately `1.25e-15`). Maximum absolute pairwise cosine change was approximately `2.78e-16`, passing the recorded `1e-12` allclose tolerances. The mean normalized-to-raw pairwise Euclidean distance ratio was `1.030494`.

**What this supports.** For this seeded synthetic matrix, the selected RMSNorm operation changed vector magnitudes and pairwise Euclidean distances while preserving pairwise cosine similarities to numerical precision.

**Limit.** This was a synthetic matrix, not a learned language-model embedding table. It says nothing about training, downstream behavior, or the semantics of learned embeddings.

**Files and artifacts.** [Experiment code and configuration](../experiments/00_embedding_geometry/); [metrics and configuration record](../results/00_embedding_geometry/metrics.json); [saved matrices](../results/00_embedding_geometry/embedding_matrices.npz); [pairwise measurements](../results/00_embedding_geometry/pairwise_geometry.npz); [distance and cosine plots](../figures/00_embedding_geometry/).

**Commit.** `f3c34aaf5953e66b5980f1615cd7b81f08b46195` — `experiment: measure embedding geometry under RMSNorm`.

## Experiment 1 checkpoint — normalization placement with a fixed table

**Question.** For one fixed embedding table, do these computational pathways produce different outputs?

- **A — post-lookup normalization:** `embedding_table[token_ids]` → RMSNorm.
- **B — pre-normalized table:** RMSNorm each row of `embedding_table` → `normalized_table[token_ids]`.

**Setup.** Both conditions used the same seeded `128 × 32` raw table from the Experiment 0 methodology, all token IDs `0` through `127`, the same RMSNorm formula and epsilon (`1e-8`), and the same fixed table. No training or Transformer was used. Equality was evaluated with `atol=1e-12` and `rtol=1e-12`.

**Observed measurements.** The maximum and mean absolute elementwise differences and maximum relative difference were all `0`. Per-token Euclidean differences, pairwise Euclidean distance differences, and pairwise cosine similarity differences were also `0`. The outputs passed the configured numerical equality check.

**Conclusion bounded to the setup.** For a fixed table and the tested RMSNorm convention, these two paths are forward-equivalent in this experiment. This is a fixed-table forward result only.

**Files and artifacts.** [Experiment code and configuration](../experiments/01_normalization_placement/); [metrics and reproducibility record](../results/01_normalization_placement/metrics.json); [saved inputs, outputs, and pairwise measurements](../results/01_normalization_placement/representations.npz).

**Commit.** `d8f88daa35ca89060a7890d2ddbc910518c4f3f4` — `experiment: test normalization placement equivalence`.

## Fixed-table forward equivalence vs. trainable-parameter behavior

These are distinct questions and must not be conflated:

1. **Fixed-table forward equivalence:** Given an unchanged embedding table and the same token IDs, do the two placements compute the same vectors? Experiment 1 tested this and found equal outputs under its stated setup and tolerance.
2. **Trainable-parameter behavior:** When parameters are optimized, do the placements produce the same gradients, update paths, parameter evolution, or learned representations? Neither completed experiment tests this. Fixed-table equality does not establish training or Transformer equivalence.

## Engineering decisions and rationale

| Decision | Rationale |
|---|---|
| Use a seeded synthetic table with small fixed dimensions in the first two experiments. | This isolates the normalization operations and keeps the initial measurements reproducible and inexpensive; it does not substitute for learned or language-model embeddings. |
| Preserve Experiment 0's RMSNorm convention, axis, and epsilon in Experiment 1. | This keeps the placement comparison controlled instead of introducing a formula change as a confound. |
| Reuse the same raw table and token IDs across Experiment 1 pathways. | Differences should reflect pathway computation, not different initialization or lookup inputs. |
| Measure pairwise geometry in Experiment 0 as well as row norms. | Row magnitudes alone do not characterize pairwise distances or angles. |
| Record configuration, runtime versions, numerical arrays, and metrics. | These records make each result inspectable and support reproducibility. |
| Separate fixed-table forward comparison from training. | Forward equivalence does not answer gradient, optimization, or trainable-parameter questions. A controlled small experiment should address those before deciding whether a larger-scale model experiment is justified. |
| Do not introduce a Transformer or external dataset in these checkpoints. | Neither is needed to answer the current fixed-table question, and introducing them now would add uncontrolled complexity. |

## Current unanswered question

When the embedding table is trainable, does the parameterization of normalization placement affect gradients, optimization dynamics, parameter evolution, or learned representations?

## Planned next experiment — Experiment 2 (not implemented)

Design a small controlled trainable-embedding experiment that compares the two placement parameterizations from the same initialization and controlled inputs. Measure gradients, optimization dynamics, parameter evolution, and resulting representations. Decide from those results whether a larger-scale model experiment is justified. The task, objective, optimizer, and analysis protocol remain to be specified before implementation. No training code or results exist yet.
