# Embedding Normalization: Does Where We Normalize Matter?

## Research question

What happens to the information and geometry of token embeddings when normalization is incorporated into the embedding representation itself, rather than applied after embedding lookup—and does normalization placement affect downstream model behavior?

## Current status

Experiments 0, 1, and 2 are complete. Experiment 0 measured geometry changes for a synthetic embedding table. Experiment 1 found fixed-table forward equivalence. Experiment 2 found numerically equal embedding-table gradients for one controlled loss. Optimization and Transformer behavior remain untested.

## Research flow

**Experiment 0: embedding geometry** → **Experiment 1: fixed-table forward equivalence** → **Experiment 2: gradients with respect to trainable E were numerically equal for the tested loss** → **open question: optimization and parameter evolution**.

See the [research log](research/RESEARCH_LOG.md) for the Mermaid investigation flow, experiment checkpoints, engineering decisions, results, and planned next question.

## Completed experiments

- [Experiment 0 — embedding geometry](experiments/00_embedding_geometry/): [metrics](results/00_embedding_geometry/metrics.json), [pairwise arrays](results/00_embedding_geometry/pairwise_geometry.npz), and [figures](figures/00_embedding_geometry/).
- [Experiment 1 — normalization placement](experiments/01_normalization_placement/): [metrics](results/01_normalization_placement/metrics.json) and [saved representations](results/01_normalization_placement/representations.npz).
- [Experiment 2 — trainable embedding gradients](experiments/02_trainable_embedding_gradients/): [metrics](results/02_trainable_embedding_gradients/metrics.json) and [saved gradients and inputs](results/02_trainable_embedding_gradients/gradients.npz).

The experiments use synthetic embeddings. Experiment 2 computes gradients but performs no optimizer updates; optimization dynamics and Transformer behavior remain untested. See the [research log](research/RESEARCH_LOG.md) for detailed methods and results.

## Paper

An Overleaf-ready LaTeX paper covering the evidence through Experiment 2 is in [paper/](paper/main.tex). It includes the existing Experiment 0 geometry figures and does not claim that the research question is solved.
