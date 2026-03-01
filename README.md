# scada-fault-prediction

Deep Learning Model for Predicting Equipment Failures Using SCADA Data
Description

This project focuses on applying deep learning models to predict industrial equipment failures using data collected from SCADA (Supervisory Control and Data Acquisition) systems.
SCADA datasets contain rich sensor signals — such as pressure, temperature, voltage, current, and flow rate — which can be leveraged to detect early signs of abnormal behavior. Early prediction helps prevent production downtime and significantly reduces maintenance costs.

Project Requirements

1. Literature Review

Study the characteristics of SCADA data and review deep learning techniques commonly used for time-series modeling, including:

RNN, LSTM, GRU

CNN and Hybrid CNN–LSTM architectures

1. Data Collection & Preprocessing

Prepare SCADA datasets by:

Cleaning and normalizing sensor variables

Handling missing or corrupted data

Extracting meaningful features for model input

1. Model Development

Implement and evaluate multiple deep learning architectures such as:

LSTM / GRU for sequential time-series forecasting

CNN or CNN–LSTM hybrids for learning complex temporal–spatial patterns

1. Model Training & Evaluation

Train models using real or simulated SCADA datasets, and evaluate performance using metrics like:

Precision

Recall

F1-Score

RMSE (Root Mean Square Error)

1. Practical Application

Develop a prototype system capable of:

Real-time failure prediction

Early-warning notifications

Integration into industrial monitoring dashboards

## Dataset Setup

To ensure the scripts can locate and process your data correctly, you need to place the raw SCADA dataset in the correct folder structure.

By default, the project expects the raw data for **Wind Farm A** to be located at:
`Dataset/raw/Wind Farm A/datasets/`

If these directories do not exist, please create them at the root of the project. Your folder structure should look like this:

```text
scada-fault-prediction/
└── Dataset/
    └── raw/
        └── Wind Farm A/
            └── datasets/
                ├── <Your raw SCADA CSV files should go here>
```

**Note:** When you run the data preprocessing scripts, the processed outputs will be automatically saved to the `Dataset/processed/` directory.
