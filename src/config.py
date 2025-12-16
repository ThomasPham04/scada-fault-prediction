"""
Configuration file for SCADA Fault Prediction project.
Contains all paths, hyperparameters, and constants.
"""

import os

# =============================================================================
# PATHS
# =============================================================================

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Wind Farm A specific paths
WIND_FARM_A_DIR = os.path.join(RAW_DATA_DIR, "Wind Farm A")
WIND_FARM_A_DATASETS = os.path.join(WIND_FARM_A_DIR, "datasets")
WIND_FARM_A_PROCESSED = os.path.join(PROCESSED_DATA_DIR, "Wind Farm A")

# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Time resolution (in minutes)
TIME_RESOLUTION = 10

# Prediction window (in hours)
PREDICTION_WINDOW_HOURS = 48

# Number of timesteps in prediction window
# 48 hours * 60 minutes / 10 minutes = 288 timesteps
SEQUENCE_LENGTH = int(PREDICTION_WINDOW_HOURS * 60 / TIME_RESOLUTION)

# Features to exclude (non-sensor columns)
EXCLUDE_COLUMNS = ['time_stamp', 'asset_id', 'id', 'train_test']

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# General
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# LSTM Model
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
DENSE_UNITS = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dirs():
    """Create necessary directories if they don't exist."""
    dirs = [
        PROCESSED_DATA_DIR,
        WIND_FARM_A_PROCESSED,
        MODELS_DIR,
        RESULTS_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    # Print configuration for verification
    print("=" * 60)
    print("SCADA Fault Prediction - Configuration")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Raw Data Directory: {RAW_DATA_DIR}")
    print(f"Processed Data Directory: {PROCESSED_DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nSequence Length: {SEQUENCE_LENGTH} timesteps ({PREDICTION_WINDOW_HOURS} hours)")
    print(f"Random Seed: {RANDOM_SEED}")
    
    # Ensure directories exist
    ensure_dirs()
    print("\nDirectories created/verified successfully!")
