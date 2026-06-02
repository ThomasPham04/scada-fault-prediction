"""Runtime defaults for sequence-classifier inference."""

from training.hyperparameters.sequence_defaults import STRIDE

DEFAULT_WINDOW_HOURS = 36
DEFAULT_STRIDE_STEPS = STRIDE
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 256
