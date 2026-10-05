"""Experiment settings, kept separate from the execution code."""

SEED = 1729
VOCABULARY_SIZE = 128
EMBEDDING_DIM = 32
NORMALIZATION_METHOD = "RMSNorm"
NORMALIZATION_AXIS = -1
NORMALIZATION_EPSILON = 1e-8
NORMALIZATION_FORMULA = "x / max(sqrt(mean(x**2, axis=-1, keepdims=True)), epsilon)"
PAIR_SELECTION = "all unique row pairs (i < j)"
OUTPUT_DIR = "results/00_embedding_geometry"
FIGURE_DIR = "figures/00_embedding_geometry"
