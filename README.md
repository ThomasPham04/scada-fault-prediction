# SCADA Fault Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.x-green.svg)](https://xgboost.readthedocs.io/)

> Deep learning and machine learning pipeline for predicting equipment failures from SCADA data.

## Description
This project applies sequence modeling and classical ML to industrial SCADA (Supervisory Control and Data Acquisition) signals such as pressure, temperature, voltage, current, and flow rate. The goal is early detection of abnormal behavior to reduce downtime and maintenance cost.

## Key Features
- Single CLI entry point via `src/main.py` for data preparation, training, and evaluation.
- Data preprocessing for missing values, standardization, and feature extraction.
- Multiple model families: deep sequence models and tree-based classifiers.
- Evaluation metrics: Precision, Recall, F1-Score, RMSE, and log-loss.

## Models Implemented
1. LSTM / GRU: stacked recurrent networks for time-series prediction.
2. AutoDecoder: reconstruction-based anomaly detection.
3. XGBoost: gradient-boosted decision trees.
4. Random Forest: ensemble classifier for robust performance.

## Dataset Setup
Place raw SCADA data in the default location (Wind Farm A):

```
scada-fault-prediction/
├── Dataset/
│   └── raw/
│       └── Wind Farm A/
│           └── datasets/
│               └── <raw SCADA CSV files>
```

Processed outputs are written to `Dataset/processed/`.

## Usage
The project uses `src/main.py` for all pipeline stages.

### 1. Data Preparation
```bash
# Prepare per-asset data (full Wind Farm A dataset)
python src/main.py prepare

# Prepare directly from a single CSV file
python src/main.py prepare --csv <path_to_csv>
```

### 2. Model Training
```bash
# Train the LSTM model on the first 10 assets
python src/main.py train --model lstm --assets 10

# Train all available architectures on a specific asset
python src/main.py train --model all --asset-ids WT_18
```

### 3. Evaluation
```bash
# Evaluate a trained Random Forest model on 10 assets
python src/main.py evaluate --model random_forest --assets 10
```

## Practical Application Target
- Real-time failure prediction from live SCADA streams.
- Early-warning notifications for maintenance engineers.
- Integration into industrial monitoring dashboards.
