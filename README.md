# Industrial Equipment Predictive Maintenance & Remaining Useful Life Prediction

> An end-to-end machine learning platform for predicting industrial
> equipment failure risk and estimating Remaining Useful Life (RUL) from
> sensor data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6.svg)](https://dvc.org/)

## Overview

Unexpected industrial equipment failures can lead to production
downtime, maintenance costs, and safety risks. This project uses machine
learning on equipment sensor data to identify degradation patterns,
estimate remaining useful life, and provide an early warning of
potential failures.

The project is designed as a **production-oriented ML system**, rather
than only a model-training notebook. It combines machine learning, an
API layer, experiment/data versioning, containerization, testing, and a
monitoring dashboard.

### Key capabilities

-   Predict **Remaining Useful Life (RUL)** of equipment
-   Estimate **equipment failure probability**
-   Process and engineer features from sensor readings
-   Compare multiple machine learning models
-   Track experiments and model versions
-   Version datasets and ML pipelines
-   Serve predictions through a REST API
-   Visualize equipment health and sensor trends
-   Containerize the application with Docker
-   Automate testing and deployment through CI/CD

------------------------------------------------------------------------

## Problem Statement

Industrial machines generate continuous sensor data such as temperature,
pressure, vibration, rotational speed, and load.

The objective of this project is to answer two practical questions:

1.  **Is the equipment likely to fail soon?**
2.  **How much useful operating life does the equipment have
    remaining?**

The system converts historical sensor observations into predictive
insights that can support condition-based and predictive maintenance.

------------------------------------------------------------------------

## System Architecture

``` text
                    ┌─────────────────────┐
                    │   Sensor Dataset    │
                    │ Temperature / RPM   │
                    │ Pressure / Vibration│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ Cleaning / Scaling  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Rolling / Statistical│
                    │ Degradation Features│
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Failure Prediction│   │ RUL Prediction  │
          │ Classification    │   │ Regression      │
          └────────┬─────────┘   └────────┬─────────┘
                   │                      │
                   └──────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ Metrics / Comparison │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              ┌──────────┐          ┌──────────┐
              │  MLflow  │          │   DVC    │
              │ Tracking │          │ Versioning│
              └────┬─────┘          └────┬─────┘
                   └──────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │   REST Prediction   │
                    │        API          │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Web Dashboard     │
                    │ Equipment Health    │
                    │ RUL / Risk / Trends │
                    └─────────────────────┘
```

------------------------------------------------------------------------

## Machine Learning Approach

### 1. Data preprocessing

The pipeline can include:

-   Missing-value handling
-   Duplicate removal
-   Outlier analysis
-   Data type validation
-   Feature scaling where required
-   Time/order-aware train-validation-test splitting
-   Prevention of data leakage

### 2. Feature engineering

Potential features include:

-   Raw sensor measurements
-   Rolling mean and standard deviation
-   Minimum/maximum sensor values
-   Sensor trends and rates of change
-   Operating-cycle statistics
-   Equipment-specific degradation indicators

### 3. Models

The project is designed to support model comparison.

#### RUL Regression

Possible models:

-   Linear Regression --- baseline
-   Random Forest Regressor
-   XGBoost Regressor
-   LightGBM Regressor
-   LSTM/GRU --- optional time-series extension

#### Failure Classification

Possible models:

-   Logistic Regression --- baseline
-   Random Forest Classifier
-   XGBoost Classifier
-   LightGBM Classifier

The final model should be selected based on validation performance and
practical considerations rather than accuracy alone.

------------------------------------------------------------------------

## Evaluation Metrics

### RUL Prediction

-   **MAE** --- Mean Absolute Error
-   **RMSE** --- Root Mean Squared Error
-   **R²** --- Coefficient of Determination

### Failure Prediction

-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Confusion Matrix
-   Precision-Recall analysis

For predictive maintenance, **recall and false-negative analysis are
especially important**, because failing to identify an upcoming
equipment failure can be more costly than generating an unnecessary
maintenance alert.

------------------------------------------------------------------------

## Technology Stack

  Layer                      Technologies
  -------------------------- ----------------------------------
  Programming                Python
  Data Processing            Pandas, NumPy
  Machine Learning           Scikit-learn, XGBoost / LightGBM
  Deep Learning (Optional)   PyTorch / TensorFlow
  Visualization              Matplotlib, Plotly
  Backend                    FastAPI
  Database                   PostgreSQL / SQLite
  Experiment Tracking        MLflow
  Data Versioning            DVC
  Containerization           Docker
  Testing                    Pytest
  CI/CD                      GitHub Actions
  Frontend                   React
  Version Control            Git & GitHub

------------------------------------------------------------------------

## Project Structure

``` text
industrial-predictive-maintenance/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_rul_modeling.ipynb
│   └── 04_failure_prediction.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train_rul.py
│   │   ├── train_failure.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   └── utils/
│       └── config.py
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       └── predictions.py
│
├── frontend/
│   └── ...
│
├── models/
│   └── .gitkeep
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
│
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md
```

------------------------------------------------------------------------

## API

The prediction service is exposed through FastAPI.

### Example endpoints

  Method   Endpoint                    Description
  -------- --------------------------- -----------------------------------
  `POST`   `/predict/rul`              Predict remaining useful life
  `POST`   `/predict/failure`          Predict failure probability
  `GET`    `/equipment/{id}`           Retrieve equipment information
  `GET`    `/equipment/{id}/history`   Retrieve sensor history
  `GET`    `/model/metrics`            Retrieve model evaluation metrics
  `GET`    `/health`                   API health check

### Example RUL request

``` json
{
  "temperature": 78.4,
  "pressure": 4.82,
  "vibration": 0.63,
  "rpm": 1820,
  "load": 72.5
}
```

### Example response

``` json
{
  "predicted_rul": 43.7,
  "unit": "cycles",
  "model_version": "rul-model-v1"
}
```

------------------------------------------------------------------------

## Dashboard

The planned dashboard provides an operational view of equipment health.

### Example information displayed

-   Overall equipment health
-   Predicted RUL
-   Failure probability
-   Current sensor readings
-   Sensor trends over time
-   Equipment history
-   High-risk equipment alerts
-   Model performance
-   Prediction history

------------------------------------------------------------------------

## MLOps

This project includes an MLOps layer to make the workflow reproducible.

### DVC

Used for:

-   Dataset versioning
-   Reproducible data pipelines
-   Tracking changes to training data

### MLflow

Used for:

-   Experiment tracking
-   Parameter logging
-   Metric logging
-   Model artifact storage
-   Model version management

### Docker

The API and supporting services can be packaged into reproducible
containers.

### GitHub Actions

CI/CD can automatically:

1.  Install dependencies
2.  Run unit tests
3.  Validate code
4.  Build Docker images
5.  Deploy the application when configured

------------------------------------------------------------------------

## Getting Started

### Prerequisites

Make sure the following are installed:

-   Python 3.10+
-   Git
-   Docker
-   DVC
-   Node.js and npm (for the React frontend)

### Clone the repository

``` bash
git clone https://github.com/<your-username>/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
```

### Create a virtual environment

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

Linux/macOS:

``` bash
source .venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

### Run the API

``` bash
uvicorn api.main:app --reload
```

The API will be available locally at:

``` text
http://127.0.0.1:8000
```

Interactive API documentation:

``` text
http://127.0.0.1:8000/docs
```

### Run with Docker

``` bash
docker build -t predictive-maintenance .
docker run -p 8000:8000 predictive-maintenance
```

------------------------------------------------------------------------

## Reproducibility

The training workflow should be reproducible using the same dataset
version, parameters, and code version.

Example workflow:

``` bash
dvc pull
dvc repro
mlflow ui
```

------------------------------------------------------------------------

## Development Roadmap

### Phase 1 --- Machine Learning

-   [x] Project definition
-   [ ] Dataset selection
-   [ ] Exploratory data analysis
-   [ ] Data preprocessing
-   [ ] Feature engineering
-   [ ] Baseline models
-   [ ] RUL model comparison
-   [ ] Failure prediction model
-   [ ] Model evaluation

### Phase 2 --- Backend

-   [ ] FastAPI application
-   [ ] Prediction endpoints
-   [ ] Request/response validation
-   [ ] Database integration
-   [ ] API testing

### Phase 3 --- MLOps

-   [ ] DVC pipeline
-   [ ] MLflow experiment tracking
-   [ ] Model versioning
-   [ ] Dockerization
-   [ ] GitHub Actions CI
-   [ ] Automated deployment

### Phase 4 --- Dashboard

-   [ ] React frontend
-   [ ] Equipment health overview
-   [ ] RUL visualization
-   [ ] Failure-risk visualization
-   [ ] Sensor trend charts
-   [ ] Alert system

### Phase 5 --- Finalization

-   [ ] End-to-end testing
-   [ ] Performance optimization
-   [ ] Documentation
-   [ ] Deployment
-   [ ] Demo video
-   [ ] Project presentation

------------------------------------------------------------------------

## Dataset

The final implementation should use a publicly available industrial
equipment/sensor dataset suitable for predictive maintenance and RUL
estimation.

The dataset source, license, preprocessing decisions, and feature
definitions will be documented here once the dataset is finalized.

> **Important:** Do not commit large raw datasets or sensitive data
> directly to Git. Use DVC or an external dataset repository where
> appropriate.

------------------------------------------------------------------------

## Results

Model results will be added after experimentation.

### RUL Model Comparison

  Model                 MAE   RMSE    R²
  ------------------- ----- ------ -----
  Linear Regression     ---    ---   ---
  Random Forest         ---    ---   ---
  XGBoost               ---    ---   ---
  LightGBM              ---    ---   ---
  LSTM (Optional)       ---    ---   ---

### Failure Classification Comparison

  Model                   Precision   Recall    F1   ROC-AUC
  --------------------- ----------- -------- ----- ---------
  Logistic Regression           ---      ---   ---       ---
  Random Forest                 ---      ---   ---       ---
  XGBoost                       ---      ---   ---       ---
  LightGBM                      ---      ---   ---       ---

------------------------------------------------------------------------

## Why This Project?

Traditional maintenance strategies often rely on:

-   Fixed maintenance schedules
-   Manual inspection
-   Reactive repairs after failure

A predictive maintenance approach can instead use equipment data to
estimate degradation and identify potential failures earlier.

This project demonstrates how an ML model can be integrated into a
complete software system---from raw data and model training to API
serving, monitoring, and deployment.

------------------------------------------------------------------------

## Future Improvements

-   Real-time sensor ingestion using Kafka or MQTT
-   Streaming predictions
-   Automated maintenance scheduling
-   Cloud deployment
-   Model drift detection
-   Automated model retraining
-   Edge-device inference
-   Advanced time-series architectures
-   Multi-equipment fleet monitoring
-   Cost-based maintenance optimization

------------------------------------------------------------------------

## Contributors

**Your Name**

-   Machine Learning
-   Backend/API
-   MLOps
-   Frontend
-   Deployment

------------------------------------------------------------------------

## License

This project is intended for educational and academic purposes.

Add an appropriate open-source license before distributing the project
publicly.
