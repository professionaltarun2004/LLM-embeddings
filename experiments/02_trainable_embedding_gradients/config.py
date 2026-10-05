"""Settings for the trainable embedding gradient comparison."""

SEED = 1729
VOCABULARY_SIZE = 128
EMBEDDING_DIM = 32
TOKEN_IDS = (3, 17, 42, 91)
NORMALIZATION_METHOD = "RMSNorm"
NORMALIZATION_AXIS = -1
NORMALIZATION_EPSILON = 1e-8
NORMALIZATION_FORMULA = "x / max(sqrt(mean(x**2, axis=-1, keepdims=True)), epsilon)"
TARGET_MIN = -0.5
TARGET_MAX = 0.5
DOWNSTREAM_COMPUTATION = "identity mapping from normalized representations"
LOSS_DEFINITION = "mean((representation - fixed_target)**2)"
RELATIVE_DIFFERENCE_DENOMINATOR_FLOOR = 1e-12
EQUALITY_ATOL = 1e-12
EQUALITY_RTOL = 1e-12
OUTPUT_DIR = "results/02_trainable_embedding_gradients"
