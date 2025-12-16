# SCADA Fault Prediction - Phase 1 Implementation

## Project Scope
- **Wind Farm:** Wind Farm A only
- **Framework:** TensorFlow/Keras
- **Task:** Binary classification (fault vs normal)
- **Prediction Window:** 48 hours ahead
- **Models:** Baseline + LSTM

## Dataset Summary
- **Events:** 22 total (12 anomalies, 10 normal)
- **Features:** 86 (temperature, wind, power, vibration sensors)
- **Timesteps per event:** ~55,000 (10-min intervals, ~385 days)
- **Fault types:** Transformer, Gearbox, Hydraulic group, Generator bearing
- **Missing values:** 2 columns have missing data

## Tasks

### Data Exploration & Understanding
- [x] Explore Wind Farm A dataset structure
- [x] Understand event_info.csv (labels)
- [x] Understand feature_description.csv
- [x] Analyze sample dataset files

### Data Preprocessing Pipeline
- [x] Load and merge SCADA data with labels
- [x] Handle missing values (forward fill / interpolation)
- [x] Normalize/standardize features
- [x] Create 48-hour sequences (288 timesteps at 10-min intervals)
- [x] Label sequences (1 if fault within 48h, 0 otherwise)
- [x] Split into train/validation/test sets
- [x] Save preprocessed data
- [x] **Implement stratified sampling to improve class balance**

### Baseline Model Development
- [/] Build simple baseline (Logistic Regression on aggregated features)
- [ ] Train baseline model
- [ ] Evaluate baseline performance

### LSTM Model Development
- [ ] Design LSTM architecture
- [ ] Implement LSTM model in Keras
- [ ] Train LSTM model with early stopping
- [ ] Evaluate LSTM performance

### Evaluation & Reporting
- [ ] Compare baseline vs LSTM
- [ ] Generate metrics (Precision, Recall, F1, Accuracy)
- [ ] Create confusion matrix
- [ ] Document findings
