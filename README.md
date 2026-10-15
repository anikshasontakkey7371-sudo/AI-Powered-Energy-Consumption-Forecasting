# AI-Powered Energy Consumption Forecasting System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/scikit--learn-RandomForest-orange.svg)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/Flask-REST%20API-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)]()

A modular, production-ready, beginner-friendly Machine Learning portfolio project designed to forecast hourly electrical power demand. This project benchmarks an ensemble **RandomForestRegressor** against a **Naive Persistence Baseline**, enforces **zero future-data leakage**, generates an autoregressive **recursive 24-hour future forecast**, and serves predictions via a **Flask REST API & Web Dashboard**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Directory Layout](#directory-layout)
- [Methodology & Leakage Prevention](#methodology--leakage-prevention)
- [Baseline vs. Machine Learning](#baseline-vs-machine-learning)
- [Quick Start & Setup](#quick-start--setup)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Flask API & Dashboard](#running-the-flask-api--dashboard)
- [Running Automated Tests](#running-automated-tests)
- [API Documentation](#api-documentation)
- [Synthetic Data Disclaimer & Ethics](#synthetic-data-disclaimer--ethics)
- [License](#license)

---

## Project Overview

Accurate short-term electricity load forecasting is vital for power grid stability, scheduling thermal generation plants, dispatching battery energy storage systems (BESS), and preventing regional brownouts.

This repository provides an end-to-end Python pipeline that:
1. Ingests hourly energy demand records (`data/energy.csv`) measured in **Megawatt-hours (MWh)**.
2. Automatically generates an authentic **synthetic hourly demonstration dataset** (8,760 hours = 1 year) if the raw dataset is missing.
3. Rigorously cleans timestamps, removes duplicates, sorts chronologically, and imputes missing observations without lookahead bias.
4. Engineers **calendar features**, **autoregressive lags**, and **rolling statistics** computed strictly on shifted past data to prevent future-data leakage.
5. Chronologically partitions historical data (**80% train / 20% test**).
6. Trains a **Naive Persistence Baseline** ($y_t = y_{t-1}$) and a **Random Forest Regressor**, evaluating both using **MAE**, **RMSE**, and **$R^2$**.
7. Executes an autoregressive **recursive 24-hour future forecast** saved to `outputs/next_24_hours.csv`.
8. Visualizes performance with publication-quality Matplotlib charts in `images/`.
9. Delivers a lightweight **Flask Web Service** with an interactive dashboard and REST endpoints.

---

## System Architecture

```
                    ┌────────────────────────────┐
                    │      data/energy.csv       │
                    │ (Auto-generated if missing)│
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │    Data Loader & Cleaner   │
                    │   (Clean, Sort, Forward-   │
                    │      Fill Imputation)      │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Leakage-Free Feature Engine│
                    │  • Calendar (hour, day, mo)│
                    │  • Lags (1, 2, 3, 24, 168) │
                    │  • Rolling (24h/168h on y-1)│
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Chronological Split (80/20)│
                    │ Train: 80%  |  Test: 20%   │
                    └──────┬──────────────┬──────┘
                           │              │
             ┌─────────────┴──┐        ┌──┴─────────────┐
             ▼                ▼        ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
     │Naive Baseline│ │Random Forest │ │ Model Metrics  │
     │  (y_t = y_t-1)│ │(100 Trees)   │ │ MAE, RMSE, R²  │
     └──────────────┘ └───────┬──────┘ └────────────────┘
                              │
                              ▼
                    ┌────────────────────────────┐
                    │ Joblib Model Serialization │
                    │ models/random_forest_model │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Recursive 24-Hour Forecast │
                    │ outputs/next_24_hours.csv  │
                    └─────────────┬──────────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            ▼                                           ▼
┌───────────────────────────────┐           ┌───────────────────────────────┐
│     Visualization Charts      │           │    Flask API & Web Dashboard  │
│ images/*.png (actual, forecast,│           │  Endpoints: /, /health,       │
│ feature_importance, residuals)│           │  /api/forecast, /api/predict  │
└───────────────────────────────┘           └───────────────────────────────┘
```

---

## Directory Layout

```
AI-Energy-Forecasting/
├── data/
│   └── energy.csv                   # Raw or synthesized hourly energy data
├── docs/
│   ├── METHODOLOGY.md               # Detailed mathematical formulation & leakage prevention
│   └── UNITS_AND_METRICS.md         # Measurement units (MWh) and metric definitions
├── images/
│   ├── actual_vs_predicted.png      # Test set comparison plot
│   ├── forecast_next_24_hours.png   # 24-hour ahead recursive forecast plot
│   ├── feature_importance.png       # Random Forest feature importance chart
│   └── residuals_distribution.png   # Residual error histogram & normal fit
├── models/
│   ├── feature_names.joblib         # Exact list of trained feature columns
│   ├── model.joblib                 # Trained RandomForestRegressor estimator
│   └── random_forest_model.joblib   # Serialized model, features, and metadata bundle
├── notebooks/
│   └── walkthrough.ipynb            # Interactive step-by-step Jupyter Notebook
├── outputs/
│   ├── metrics.csv                  # Baseline vs. Random Forest evaluation scores
│   ├── next_24_hours.csv            # 24-hour autoregressive future forecast
│   ├── predictions.csv              # Test set actuals, predictions, and residuals
│   └── *.png                        # Dual-saved diagnostic plots
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── data_loader.py               # Ingestion, validation, cleaning, and synthesis
│   ├── feature_engineering.py       # Calendar, lag, and rolling features (no leakage)
│   ├── forecasting.py               # Autoregressive multi-step recursive forecaster
│   ├── model.py                     # Chronological split, baseline, RF model, metrics
│   └── visualization.py             # Publication-ready Matplotlib chart generation
├── tests/
│   ├── __init__.py
│   ├── test_app.py                  # Flask endpoint and route tests
│   ├── test_data_loader.py          # Data ingestion and cleaning tests
│   ├── test_feature_engineering.py  # Leakage-prevention and lag tests
│   ├── test_forecasting.py          # Recursive forecast integrity tests
│   └── test_model.py                # Model training, splitting, and joblib tests
├── .gitignore                       # Standard Python ignore rules
├── app.py                           # Flask REST API server and HTML dashboard
├── main.py                          # Full pipeline orchestration script
├── README.md                        # Documentation and guide
└── requirements.txt                 # Project dependencies
```

---

## Methodology & Leakage Prevention

A critical issue in time-series machine learning is **future-data leakage (lookahead bias)**:

1. **Chronological Partitioning:**
   - Shuffling time-series observations leaks future observations into past training sets.
   - We split chronologically: the first 80% of rows are training data, and the remaining 20% are held-out test data.
   - We enforce: $\max(T_{\text{train}}) < \min(T_{\text{test}})$.

2. **Shifted Rolling Aggregations:**
   - A rolling average that includes observation $y_t$ directly leaks the target into the features.
   - We explicitly shift the target by 1 hour before computing any rolling statistics:
     ```python
     df['rolling_mean_24'] = df['Energy'].shift(1).rolling(window=24).mean()
     df['rolling_std_24']  = df['Energy'].shift(1).rolling(window=24).std()
     df['rolling_mean_168'] = df['Energy'].shift(1).rolling(window=168).mean()
     ```
   - This guarantees that for row $t$, only observations up to $t-1$ are accessed.

3. **Autoregressive Lags:**
   - Lags at $t-1$, $t-2$, $t-3$, $t-24$, $t-48$, and $t-168$ represent past observations.
   - Initial rows containing `NaN` during the 168-hour warm-up window are cleanly dropped before training.

---

## Baseline vs. Machine Learning

In real-world power systems, complex models must justify their compute and maintenance costs by outperforming the **Naive Persistence Baseline**:

$$\hat{y}_{t}^{\text{baseline}} = y_{t-1}$$

Both models are evaluated on identical held-out test data using:
- **MAE (Mean Absolute Error):** Average magnitude of absolute error in Megawatt-hours (MWh).
- **RMSE (Root Mean Squared Error):** Square root of the mean squared errors, penalizing severe spikes.
- **$R^2$ (Coefficient of Determination):** Proportion of variance captured by the model.

---

## Quick Start & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 2. Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/AI-Energy-Forecasting.git
cd AI-Energy-Forecasting

# Create a local virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (cmd.exe):
.\.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

To run the complete machine learning pipeline (data ingestion, feature engineering, model training, evaluation, recursive forecasting, and chart generation):

```bash
python main.py
```

### Sample Terminal Output:
```
======================================================================
   AI-POWERED ENERGY CONSUMPTION FORECASTING SYSTEM
   End-to-End Machine Learning & Evaluation Pipeline
======================================================================

[NOTE] No existing 'data/energy.csv' found.
       Generated synthetic hourly demonstration dataset (8,760 hours = 1 year).
       Unit: Megawatt-hours (MWh). Clearly labeled for portfolio evaluation.

-----------------------------------------------------------------
                MODEL EVALUATION COMPARISON
-----------------------------------------------------------------
Model Name                       | MAE (MWh)  | RMSE (MWh) | R² Score
-----------------------------------------------------------------
Naive Baseline (Previous Hour)   | 3.984      | 5.036      | 0.7583  
RandomForestRegressor            | 2.729      | 3.454      | 0.8863  
-----------------------------------------------------------------
Random Forest Error Reduction vs Baseline: MAE by 31.5% | RMSE by 31.4%

 Artifacts generated:
  • Trained Model    : models/random_forest_model.joblib
  • Feature Names    : models/feature_names.joblib
  • 24-Hour Forecast : outputs/next_24_hours.csv
  • Test Predictions : outputs/predictions.csv
  • Evaluation Metrics: outputs/metrics.csv
  • Diagnostic Charts: images/ and outputs/ (actual_vs_predicted, 24h_forecast, feature_importance, residuals)
```

---

## Running the Flask API & Dashboard

Launch the Flask web service:

```bash
python app.py
```

Open your browser at **`http://127.0.0.1:5000`** to view the interactive dashboard displaying model status, live metrics, and the 24-hour forecast table.

---

## Running Automated Tests

A comprehensive suite of automated tests verifies every module:

```bash
pytest -v
```

Tests cover:
- Synthetic data generation and schema integrity
- Missing value imputation and timestamp gap continuity reindexing
- Absence of lookahead bias in rolling statistics and lags
- Strict chronological separation of training and test sets
- Model training, persistence baseline, and Joblib artifact serialization
- Autoregressive recursive 24-hour continuity and lag propagation
- Flask API routes, error handlers, and prediction endpoints

---

## API Documentation

### 1. Web Dashboard
- **Route:** `GET /`
- **Description:** Renders an HTML dashboard with KPI cards and the 24-hour forecast table.

### 2. Health & Status
- **Route:** `GET /health`
- **Sample Response:**
  ```json
  {
    "data_source_mode": "Synthetic Demonstration (Simulated)",
    "forecast_available": true,
    "model_loaded": true,
    "model_type": "RandomForestRegressor",
    "service": "AI-Powered Energy Consumption Forecasting System",
    "status": "healthy",
    "unit": "MWh"
  }
  ```

### 3. Fetch 24-Hour Forecast
- **Route:** `GET /api/forecast`
- **Sample Response:**
  ```json
  {
    "forecast_horizon_hours": 24,
    "unit": "MWh",
    "records": [
      {
        "Datetime": "2024-12-31 00:00:00",
        "Forecast_Energy": 66.83,
        "Generated_At": "2026-10-15 00:14:15",
        "Model": "RandomForestRegressor",
        "Unit": "MWh",
        "Data_Source": "Synthetic Demonstration Model (Educational Simulation)"
      }
    ]
  }
  ```

### 4. Fetch Evaluation Metrics
- **Route:** `GET /api/metrics`
- **Sample Response:**
  ```json
  {
    "metrics": [
      {
        "model_name": "Naive Baseline (Previous Hour)",
        "mae": 3.984,
        "rmse": 5.036,
        "r2_score": 0.7583,
        "sample_size": 1719,
        "unit": "MWh"
      },
      {
        "model_name": "RandomForestRegressor",
        "mae": 2.729,
        "rmse": 3.454,
        "r2_score": 0.8863,
        "sample_size": 1719,
        "unit": "MWh"
      }
    ]
  }
  ```

### 5. On-Demand Single-Step Prediction
- **Route:** `POST /api/predict`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "features": {
      "hour": 14,
      "dayofweek": 2,
      "month": 6,
      "is_weekend": 0,
      "lag_1": 55.2,
      "lag_2": 54.1,
      "lag_3": 52.8,
      "lag_24": 53.9,
      "lag_48": 54.0,
      "lag_168": 56.1,
      "rolling_mean_24": 51.5,
      "rolling_std_24": 4.2,
      "rolling_mean_168": 50.8
    }
  }
  ```
- **Response:**
  ```json
  {
    "model": "RandomForestRegressor",
    "prediction": 56.14,
    "unit": "MWh"
  }
  ```

---

## Synthetic Data Disclaimer & Ethics

- **Benchmarking & Educational Purpose:** In the absence of an external utility dataset, this project procedurally generates a realistic hourly time series reflecting diurnal rhythms, weekend commercial dips, and seasonal temperature swings.
- **No False Claims:** Synthetic benchmark results are explicitly labeled across all logs, API responses, CSV exports, and UI elements.
- **Real-World Deployment:** For commercial deployment, connect `data/energy.csv` to validated smart meter or ISO/RTO telemetry feeds (e.g., PJM, CAISO, ERCOT, or ENTSO-E).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
