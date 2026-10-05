# Embedding Normalization: Does Where We Normalize Matter?

## Research question

What happens to token embedding information and geometry when normalization is incorporated into the embedding representation itself rather than applied after embedding lookup, and does normalization placement affect downstream model behavior?

## Current status

Experiment 0 is implemented as a synthetic embedding geometry baseline. Experiment 1 has not started. No Transformer is implemented.

## Experiment philosophy

Keep experiments small, explicit, and interpretable. Separate configuration from implementation, use controlled comparisons, save raw numerical outputs and plots as distinct artifacts, and avoid adding frameworks or features before they are needed.

## Reproducibility requirements

Record configuration and use an explicit deterministic random seed for every experiment. Keep dependencies minimal, preserve numerical outputs and figures, and make each experiment runnable from the repository root with its declared requirements installed.

## Experiment 0

Experiment `00_embedding_geometry` creates a seeded synthetic matrix and compares its rows with the same rows after RMSNorm. It records row and coordinate summaries, exact pairwise Euclidean distances and normalized-to-raw distance ratios for all unique row pairs, plus pairwise cosine similarities before and after normalization. The outputs include the normalization convention and runtime versions; numerical arrays and metrics are saved separately from distance and cosine comparison plots.

This experiment can describe norm changes and pairwise geometry changes for this matrix under this row-wise scaling operation. It cannot establish what happens to information in learned embeddings, how a language model behaves downstream, or whether normalization placement affects training. The matrix is synthetic and is not a learned language-model embedding table.
