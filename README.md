# Embedding Normalization: Does Where We Normalize Matter?

## Research question

What happens to token embedding information and geometry when normalization is incorporated into the embedding representation itself rather than applied after embedding lookup, and does normalization placement affect downstream model behavior?

## Current status

Repository bootstrap only. Experiment `00_embedding_geometry` is an executable skeleton; no research conclusions have been produced, and no Transformer is implemented.

## Experiment philosophy

Keep experiments small, explicit, and interpretable. Separate configuration from implementation, use controlled comparisons, save raw numerical outputs and plots as distinct artifacts, and avoid adding frameworks or features before they are needed.

## Reproducibility requirements

Record configuration and use an explicit deterministic random seed for every experiment. Keep dependencies minimal, preserve numerical outputs and figures, and make each experiment runnable from the repository root with its declared requirements installed.
