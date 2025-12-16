# Phase 1 Implementation Plan: SCADA Fault Prediction

## Goal

Build a binary classification system to predict wind turbine faults 48 hours in advance using SCADA sensor data from Wind Farm A. Implement both a baseline model and an LSTM model using TensorFlow/Keras.

## User Review Required

> [!IMPORTANT]
> **48-Hour Prediction Window**: The model will predict if a fault will occur within the next 48 hours (288 timesteps at 10-minute intervals). This means we'll create sliding windows and label them as 1 if a fault occurs within 48 hours, 0 otherwise.

> [!IMPORTANT]
> **Data Imbalance**: Dataset has 12 anomaly events vs 10 normal events. Within each event, most timesteps will be labeled as 0 (normal), with only the 48-hour window before faults labeled as 1. We'll need to handle class imbalance.

> [!IMPORTANT]
> **Missing Values**: 2 columns have missing values. We'll use forward fill as the primary strategy for time-series data.

## Proposed Changes

### Data Preprocessing

#### [NEW] [preprocess_data.py](file:///d:/Final%20Project/scada-fault-prediction/src/preprocessing_data/preprocess_data.py)

Main preprocessing pipeline that will:
- Load all 22 event CSV files from Wind Farm A
- Merge with event_info.csv to get labels and fault timestamps
- Handle missing values using forward fill
- Normalize features using StandardScaler (fit on train, transform on train/val/test)
- Create sliding windows of 288 timesteps (48 hours)
- Label each window: 1 if fault occurs within next 48 hours, 0 otherwise
- Split data: 70% train, 15% validation, 15% test (stratified by event type)
- Save preprocessed data as `.npz` files

**Key functions:**
- `load_event_data(event_id)` - Load single event CSV
- `create_labels(timestamps, event_info)` - Create binary labels based on 48h window
- `create_sequences(data, labels, window_size=288)` - Create sliding windows
- `preprocess_pipeline()` - Main orchestration function

#### [MODIFY] [requirements.txt](file:///d:/Final%20Project/scada-fault-prediction/requirements.txt)

Add necessary dependencies:
```
tensorflow>=2.13.0
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

### Model Development

#### [NEW] [baseline_model.py](file:///d:/Final%20Project/scada-fault-prediction/src/model/baseline_model.py)

Simple baseline using aggregated statistics:
- Extract statistical features from each 48-hour window (mean, std, min, max)
- Train Logistic Regression classifier
- Evaluate on test set
- Save model and results

**Purpose:** Establish baseline performance to compare against LSTM

#### [NEW] [lstm_model.py](file:///d:/Final%20Project/scada-fault-prediction/src/model/lstm_model.py)

LSTM model architecture:
```
Input: (288 timesteps, 86 features)
├── LSTM(64 units, return_sequences=True)
├── Dropout(0.2)
├── LSTM(32 units)
├── Dropout(0.2)
├── Dense(16, activation='relu')
├── Dense(1, activation='sigmoid')
```

**Training configuration:**
- Loss: Binary crossentropy with class weights (to handle imbalance)
- Optimizer: Adam (lr=0.001)
- Metrics: Accuracy, Precision, Recall, AUC
- Early stopping: patience=10, monitor='val_loss'
- Batch size: 32
- Epochs: 50 (with early stopping)

---

### Evaluation & Utilities

#### [NEW] [evaluate.py](file:///d:/Final%20Project/scada-fault-prediction/src/model/evaluate.py)

Evaluation utilities:
- Load trained models
- Generate predictions on test set
- Calculate metrics: Precision, Recall, F1, Accuracy, AUC
- Plot confusion matrix
- Plot ROC curve
- Save results to JSON

#### [NEW] [config.py](file:///d:/Final%20Project/scada-fault-prediction/src/config.py)

Configuration file with constants:
- Data paths
- Model hyperparameters
- Preprocessing parameters
- Random seed for reproducibility

---

## Verification Plan

### Automated Tests

#### Data Preprocessing Verification
```bash
# Run preprocessing
python src/preprocessing_data/preprocess_data.py

# Verify outputs exist
ls Dataset/processed/Wind\ Farm\ A/
# Expected: X_train.npz, X_val.npz, X_test.npz, y_train.npz, y_val.npz, y_test.npz, scaler.pkl
```

**Validation checks:**
- Verify sequence shape: (n_samples, 288, 86)
- Verify label distribution: check class imbalance ratio
- Verify no data leakage: train/val/test from different events
- Verify normalization: mean ≈ 0, std ≈ 1 for train set

#### Baseline Model Training
```bash
python src/model/baseline_model.py
```

**Expected output:**
- Model saved to `models/baseline_lr.pkl`
- Metrics printed: Accuracy, Precision, Recall, F1
- Baseline should achieve >50% accuracy (better than random)

#### LSTM Model Training
```bash
python src/model/lstm_model.py
```

**Expected output:**
- Model saved to `models/lstm_model.h5`
- Training history plot saved
- Metrics printed: Accuracy, Precision, Recall, F1, AUC
- LSTM should outperform baseline

### Manual Verification

#### Visual Inspection
```bash
python src/model/evaluate.py
```

**User should verify:**
1. Confusion matrix shows reasonable predictions
2. ROC curve shows AUC > 0.7 (target)
3. Precision/Recall tradeoff is acceptable for early warning system
4. Training curves show no overfitting (val_loss doesn't diverge from train_loss)

#### Results Comparison
- Compare baseline vs LSTM metrics
- LSTM should show improvement in Recall (critical for fault detection)
- Document findings in experimental report

---

## Success Criteria

1. ✅ Preprocessing pipeline runs without errors
2. ✅ Data shapes are correct (288 timesteps, 86 features)
3. ✅ Baseline model achieves >60% accuracy
4. ✅ LSTM model achieves >70% accuracy and >0.7 AUC
5. ✅ LSTM outperforms baseline in Recall (minimize false negatives)
6. ✅ All models and results saved for reporting
