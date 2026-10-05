# Embedding Normalization: Does Where We Normalize Matter?

## Research question

What happens to the information and geometry of token embeddings when normalization is incorporated into the embedding representation itself, rather than applied after embedding lookup—and does normalization placement affect downstream model behavior?

## Current status

Experiments 0 and 1 are complete. Experiment 0 measured geometry changes for a synthetic embedding table. Experiment 1 found fixed-table forward equivalence for the two tested RMSNorm placements. Trainable-parameter behavior remains untested; Experiment 2 is planned but has not started.

## Research flow

**Experiment 0: embedding geometry** → cosine similarities were preserved to numerical precision while Euclidean distances changed → **Experiment 1: normalization placement** → fixed-table forward outputs were numerically equal → **open question: trainable-parameter behavior**.

See the [research log](research/RESEARCH_LOG.md) for the Mermaid investigation flow, experiment checkpoints, engineering decisions, results, and planned next question.

## Completed experiments

- [Experiment 0 — embedding geometry](experiments/00_embedding_geometry/): [metrics](results/00_embedding_geometry/metrics.json), [pairwise arrays](results/00_embedding_geometry/pairwise_geometry.npz), and [figures](figures/00_embedding_geometry/).
- [Experiment 1 — normalization placement](experiments/01_normalization_placement/): [metrics](results/01_normalization_placement/metrics.json) and [saved representations](results/01_normalization_placement/representations.npz).

Both experiments use synthetic embeddings. Neither evaluates training or Transformer behavior.
