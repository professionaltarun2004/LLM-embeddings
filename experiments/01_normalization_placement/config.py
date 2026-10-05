"""Settings for the fixed-table normalization-placement comparison."""

SEED = 1729
VOCABULARY_SIZE = 128
EMBEDDING_DIM = 32
TOKEN_IDS = tuple(range(VOCABULARY_SIZE))
NORMALIZATION_METHOD = "RMSNorm"
NORMALIZATION_AXIS = -1
NORMALIZATION_EPSILON = 1e-8
NORMALIZATION_FORMULA = "x / max(sqrt(mean(x**2, axis=-1, keepdims=True)), epsilon)"
EQUALITY_ATOL = 1e-12
EQUALITY_RTOL = 1e-12
RELATIVE_DIFFERENCE_DENOMINATOR_FLOOR = 1e-12
OUTPUT_DIR = "results/01_normalization_placement"
