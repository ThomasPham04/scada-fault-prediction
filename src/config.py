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
NBM_PROCESSED_DIR = os.path.join(WIND_FARM_A_PROCESSED, "NBM_7day")

# Experiment outputs
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

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
# NORMAL BEHAVIOR MODEL (NBM) PARAMETERS
# =============================================================================

# NBM Window Configuration
# nhun
NBM_WINDOW_DAYS = 3  # Changed from 14 to 7 days
NBM_WINDOW_SIZE = int(NBM_WINDOW_DAYS * 24 * 60 / TIME_RESOLUTION)  # 1008 timesteps (was 2016)
NBM_STRIDE = 72  # 6 hours stride (was 72 - 12 hours)
# Normal Data Filtering Criteria
NBM_CUT_IN_WIND_SPEED = 4.0  # m/s
NBM_MIN_POWER = 0.0  # kW
NBM_NORMAL_STATUS = [0]  # Only status_type == 0 (normal operation)

# Feature Groups for NBM
NBM_TEMPERATURE_FEATURES = [
    'sensor_0_avg',   # Ambient temperature
    'sensor_6_avg',   # Hub controller temp
    'sensor_7_avg',   # Nacelle controller temp
    'sensor_8_avg',   # VCS choke coils temp
    'sensor_9_avg',   # VCP-board temp
    'sensor_10_avg',  # VCS cooling water temp
    'sensor_11_avg',  # Gearbox bearing temp (CRITICAL)
    'sensor_12_avg',  # Gearbox oil temp (CRITICAL)
    'sensor_13_avg',  # Generator bearing DE (CRITICAL)
    'sensor_14_avg',  # Generator bearing NDE (CRITICAL)
    'sensor_15_avg',  # Generator stator L1
    'sensor_16_avg',  # Generator stator L2
    'sensor_17_avg',  # Generator stator L3
    'sensor_19_avg',  # Split ring chamber temp
    'sensor_20_avg',  # Busbar section temp
    'sensor_21_avg',  # IGBT grid side inverter temp
    'sensor_35_avg',  # IGBT rotor side inverter L1
    'sensor_36_avg',  # IGBT rotor side inverter L2
    'sensor_37_avg',  # IGBT rotor side inverter L3
    'sensor_38_avg',  # HV transformer L1 (CRITICAL)
    'sensor_39_avg',  # HV transformer L2 (CRITICAL)
    'sensor_40_avg',  # HV transformer L3 (CRITICAL)
    'sensor_41_avg',  # Hydraulic oil temp (CRITICAL)
    'sensor_43_avg',  # Nacelle temperature
    'sensor_53_avg',  # Nose cone temperature
]

NBM_RPM_FEATURES = [
    'sensor_18_avg',  # Generator RPM
    'sensor_18_max',
    'sensor_18_min',
    'sensor_18_std',
    'sensor_52_avg',  # Rotor RPM
    'sensor_52_max',
    'sensor_52_min',
    'sensor_52_std',
]

NBM_ELECTRICAL_FEATURES = [
    'sensor_23_avg',  # Current L1
    'sensor_24_avg',  # Current L2
    'sensor_25_avg',  # Current L3
    'sensor_32_avg',  # Voltage L1
    'sensor_33_avg',  # Voltage L2
    'sensor_34_avg',  # Voltage L3
    'sensor_26_avg',  # Grid frequency
    'sensor_22_avg',  # Phase displacement
]

NBM_WIND_POWER_FEATURES = [
    'wind_speed_3_avg',
    'wind_speed_3_max',
    'wind_speed_3_min',
    'wind_speed_3_std',
    'wind_speed_4_avg',  # Estimated windspeed
    'power_29_avg',      # Possible grid active power
    'power_29_max',
    'power_29_min',
    'power_29_std',
    'power_30_avg',      # Grid power
    'power_30_max',
    'power_30_min',
    'power_30_std',
    'reactive_power_27_avg',
    'reactive_power_27_max',
    'reactive_power_27_min',
    'reactive_power_27_std',
    'reactive_power_28_avg',
    'reactive_power_28_max',
    'reactive_power_28_min',
    'reactive_power_28_std',
    'sensor_31_avg',  # Grid reactive power
    'sensor_31_max',
    'sensor_31_min',
    'sensor_31_std',
]

# Angle features (will be converted to sin/cos)
NBM_ANGLE_FEATURES = [
    'sensor_1_avg',   # Wind absolute direction
    'sensor_2_avg',   # Wind relative direction
    'sensor_5_avg',   # Pitch angle
    'sensor_5_max',
    'sensor_5_min',
    'sensor_5_std',
    'sensor_42_avg',  # Nacelle direction
]

# Cumulative energy features (will be DROPPED - not useful for NBM)
# These features are in Wh (Watt-hour) and VArh (Volt-Ampere reactive hour) units,
# representing cumulative energy rather than instantaneous measurements.
# NBM requires real-time state indicators, not accumulated counters.
# We already have instantaneous power features: power_29, power_30, sensor_31 (kW, kVAr)
# NOTE: These sensors don't have '_avg' suffix in the CSV files
NBM_COUNTER_FEATURES = [
    'sensor_44',  # Active power - generator disconnected (Wh)
    'sensor_45',  # Active power - generator delta (Wh)
    'sensor_46',  # Active power - generator star (Wh)
    'sensor_47',  # Reactive power - generator disconnected (VArh)
    'sensor_48',  # Reactive power - generator delta (VArh)
    'sensor_49',  # Reactive power - generator star (VArh)
    'sensor_50',  # Total active power (Wh)
    'sensor_51',  # Total reactive power (VArh)
]

# Combine all feature groups (excluding angles and counters - will be processed separately)
NBM_FEATURE_COLUMNS = (
    NBM_TEMPERATURE_FEATURES +
    NBM_RPM_FEATURES +
    NBM_ELECTRICAL_FEATURES +
    NBM_WIND_POWER_FEATURES
)

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

# NBM LSTM Prediction Model
NBM_LSTM_UNITS_1 = 128
NBM_LSTM_UNITS_2 = 64
NBM_DROPOUT_RATE = 0.2
NBM_DENSE_UNITS = 64
NBM_LEARNING_RATE = 0.0001
NBM_BATCH_SIZE = 32
NBM_EPOCHS = 30
NBM_EARLY_STOPPING_PATIENCE = 5

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
