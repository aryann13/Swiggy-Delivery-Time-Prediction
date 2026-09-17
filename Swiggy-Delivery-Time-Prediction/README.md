<div align="center">

# 🛵 Swiggy Delivery Time Prediction

**End-to-end Machine Learning pipeline to predict food delivery time, built with production-grade MLOps tooling.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006600?logo=xgboost)](https://xgboost.readthedocs.io)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-DagsHub-0194E2?logo=mlflow&logoColor=white)](https://dagshub.com/aryann13/Swiggy-Delivery-Time-Prediction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Live Demo](#-quick-start) · [Architecture](#-architecture) · [Model Details](#-model-architecture) · [API Docs](#-api-reference) · [Notebooks](#-experimental-notebooks)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Model Architecture](#-model-architecture)
- [DVC Pipeline](#-dvc-pipeline)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Experimental Notebooks](#-experimental-notebooks)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

This project predicts **food delivery time (in minutes)** for Swiggy — India's leading food delivery platform. Given order details like restaurant location, delivery location, weather, traffic density, and vehicle type, the model estimates how long the delivery will take.

### What Makes This Project Stand Out

| Aspect | Implementation |
| :--- | :--- |
| **Model** | `TransformedTargetRegressor` wrapping a `StackingRegressor` (Random Forest + XGBoost) with `PowerTransformer` on the target |
| **Pipeline** | Fully reproducible 6-stage DVC pipeline (`clean → prepare → preprocess → train → evaluate → register`) |
| **Experiment Tracking** | MLflow on DagsHub — metrics, parameters, datasets, and model artifacts are all versioned |
| **Model Registry** | Model registered and promoted to `Staging` stage in MLflow Model Registry on DagsHub |
| **Serving** | FastAPI REST API with Pydantic data contracts and Jinja2-rendered dark-themed web UI |
| **Data Contracts** | Pydantic schema enforces strict input validation at the API gateway — no garbage in, no garbage out |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DVC Pipeline (dvc.yaml)                        │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌───────┐   ┌──────┐│
│  │  Data     │──▶│  Data    │──▶│    Data      │──▶│ Train │──▶│Eval  ││
│  │ Cleaning  │   │  Prep    │   │ Preprocessing│   │       │   │      ││
│  └──────────┘   └──────────┘   └──────────────┘   └───────┘   └──┬───┘│
│   Haversine      80/20 Split    MinMax + OHE +      Stacking     │     │
│   Distance       (stratified)   OrdinalEncoder      Regressor    │     │
│   Extraction                                        (RF+XGB)     │     │
│                                                                   │     │
│                                                         ┌─────────▼───┐│
│                                                         │  Register   ││
│                                                         │   Model     ││
│                                                         └──────┬──────┘│
└─────────────────────────────────────────────────────────────────┼───────┘
                                                                  │
                    ┌─────────────────────────────────────────────▼───────┐
                    │              DagsHub / MLflow Registry               │
                    │  • Metrics (MAE, R², CV scores)                      │
                    │  • Parameters (RF + XGBoost hyperparams)             │
                    │  • Artifacts (model.joblib, preprocessor.joblib)     │
                    │  • Model Stage: Staging                              │
                    └─────────────────────────────────────────────┬───────┘
                                                                  │
                    ┌─────────────────────────────────────────────▼───────┐
                    │              FastAPI Application (app.py)            │
                    │                                                      │
                    │  GET  /        → Jinja2 Web UI (Dark Theme)          │
                    │  POST /predict → JSON prediction endpoint            │
                    │  GET  /docs    → Swagger/OpenAPI documentation        │
                    │                                                      │
                    │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │
                    │  │  Pydantic  │─▶│ Data Cleaning│─▶│  model_pipe  │  │
                    │  │  Validation│  │  (features)  │  │ (preprocess  │  │
                    │  │            │  │              │  │  + regressor)│  │
                    │  └────────────┘  └──────────────┘  └──────────────┘  │
                    └──────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture

```
TransformedTargetRegressor
├── transformer: PowerTransformer()               # Normalizes skewed delivery times
│
└── regressor: StackingRegressor (5-fold CV)
    │
    ├── Base Learner 1: RandomForestRegressor
    │   ├── n_estimators: 296
    │   ├── max_depth: 14
    │   ├── min_samples_split: 6
    │   ├── min_samples_leaf: 2
    │   └── max_samples: 0.6447
    │
    ├── Base Learner 2: XGBRegressor
    │   ├── n_estimators: 232
    │   ├── max_depth: 15
    │   ├── learning_rate: 0.1407
    │   ├── subsample: 0.7424
    │   ├── min_child_weight: 3
    │   ├── gamma: 1.1254
    │   └── reg_lambda: 5.8792
    │
    └── Meta-Learner: LinearRegression()          # Blends RF + XGB predictions
```

### Why This Architecture?

1. **Stacking Regressor** — Combines the variance-reduction strength of Random Forest with the gradient-boosting precision of XGBoost. The `LinearRegression` meta-learner learns the optimal weighting between them via 5-fold cross-validation.

2. **TransformedTargetRegressor** — Delivery times are right-skewed (many 20-min orders, few 60+ min). The `PowerTransformer` (Yeo-Johnson) normalizes the target distribution during training, improving the model's ability to predict both short and long deliveries accurately.

### Feature Engineering

| Feature | Source | Transformation |
| :--- | :--- | :--- |
| `distance` | Restaurant & delivery coordinates | Haversine formula (km) |
| `distance_type` | `distance` | Binned: short / medium / long / very_long |
| `pickup_time_minutes` | Order time → pickup time | Time delta in minutes |
| `order_time_of_day` | Order hour | Categorized: morning / afternoon / evening / night / after_midnight |
| `is_weekend` | Order date | Binary: Saturday/Sunday = 1 |
| `weather` | Raw conditions string | Cleaned & lowercased |
| `traffic` | Raw density string | Ordinal encoded: low → medium → high → jam |

### Preprocessing Pipeline (ColumnTransformer)

| Column Type | Columns | Transformer |
| :--- | :--- | :--- |
| **Numerical** | `age`, `ratings`, `pickup_time_minutes`, `distance` | `MinMaxScaler` |
| **Nominal Categorical** | `weather`, `type_of_order`, `type_of_vehicle`, `festival`, `city_type`, `is_weekend`, `order_time_of_day` | `OneHotEncoder` (drop first) |
| **Ordinal Categorical** | `traffic`, `distance_type` | `OrdinalEncoder` (ordered) |

---

## 🔄 DVC Pipeline

The entire ML workflow is orchestrated via [DVC](https://dvc.org) — making every experiment fully reproducible.

```yaml
# dvc.yaml — 6-stage pipeline
stages:
  data_cleaning:      # Raw CSV → cleaned features (Haversine, time extraction)
  data_preparation:   # 80/20 train-test split
  data_preprocessing: # ColumnTransformer (MinMax + OHE + Ordinal) → preprocessor.joblib
  train:              # StackingRegressor(RF + XGB) → model.joblib
  evaluation:         # MAE, R², CV scores → logged to MLflow on DagsHub
  register_model:     # Push model to MLflow Model Registry → Staging
```

**Reproduce the entire pipeline:**
```bash
dvc repro
```

**View pipeline DAG:**
```bash
dvc dag
```

```
    +----------------+
    | data_cleaning  |
    +----------------+
            *
            *
            *
   +-----------------+
   | data_preparation|
   +-----------------+
            *
            *
            *
+---------------------+
| data_preprocessing  |
+---------------------+
            *
            *
            *
       +---------+
       |  train  |
       +---------+
            *
            *
            *
     +------------+
     | evaluation |
     +------------+
            *
            *
            *
   +----------------+
   | register_model |
   +----------------+
```

---

## 📁 Project Structure

```
Swiggy-Delivery-Time-Prediction/
│
├── app.py                          # FastAPI application (API + Web UI)
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # Pipeline lock file (reproducibility)
├── params.yaml                     # Hyperparameters for RF and XGBoost
├── run_information.json            # MLflow run ID and model registry info
├── requirements.txt                # Python dependencies
│
├── src/                            # Production source code
│   ├── data/
│   │   ├── data_cleaning.py        # Stage 1: Raw data → cleaned features
│   │   └── data_preparation.py     # Stage 2: Train/test split
│   ├── features/
│   │   └── data_preprocessing.py   # Stage 3: ColumnTransformer pipeline
│   └── models/
│       ├── train.py                # Stage 4: Stacking Regressor training
│       ├── evaluation.py           # Stage 5: Metrics + MLflow logging
│       └── register_model.py       # Stage 6: MLflow Model Registry
│
├── scripts/
│   └── data_clean_utils.py         # Shared cleaning utilities (Haversine, time features)
│
├── models/                         # Serialized model artifacts
│   ├── model.joblib                # TransformedTargetRegressor (complete model)
│   ├── preprocessor.joblib         # Fitted ColumnTransformer
│   └── power_transformer.joblib    # Fitted PowerTransformer
│
├── notebooks/                      # Experimental Jupyter notebooks
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   ├── Exp1dropVSimpute.ipynb      # Experiment 1: Drop vs Impute missing values
│   ├── Exp2missingIndicator.ipynb  # Experiment 2: Missing indicator features
│   ├── Exp3modelSelectionfeatureeng.ipynb  # Experiment 3: Model selection
│   └── final_model_building.ipynb  # Final model with stacking + hypertuning
│
├── templates/
│   └── index.html                  # Jinja2 template (dark-themed prediction UI)
│
├── static/
│   ├── style.css                   # Glassmorphism dark theme
│   └── script.js                   # Frontend form logic + API integration
│
└── data/                           # Data directory (managed by DVC)
    ├── raw/swiggy.csv              # Original dataset
    ├── cleaned/swiggy_cleaned.csv  # After data_cleaning stage
    ├── interim/                    # Train/test splits
    └── processed/                  # Transformed features
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/aryann13/Swiggy-Delivery-Time-Prediction.git
cd Swiggy-Delivery-Time-Prediction/Swiggy-Delivery-Time-Prediction
```

### 2. Install Dependencies

```bash
pip install pandas scikit-learn xgboost fastapi uvicorn jinja2 python-multipart mlflow dagshub joblib pyyaml python-dotenv
```

### 3. Set Up Environment Variables

Create a `.env` file with your DagsHub credentials (required for MLflow model loading):

```env
DAGSHUB_USER_TOKEN=your_dagshub_token_here
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Browser

- **Web UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000) — fill in the form and get a prediction
- **API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — interactive Swagger UI

---

## 📡 API Reference

### `GET /`

Renders the prediction web UI (Jinja2 template with dark glassmorphism theme).

### `POST /predict`

Returns the predicted delivery time in minutes.

**Request Body:**

```json
{
  "ID": "5632A0",
  "Delivery_person_ID": "INDORES13DEL02",
  "Delivery_person_Age": "29",
  "Delivery_person_Ratings": "4.5",
  "Restaurant_latitude": 22.745049,
  "Restaurant_longitude": 75.892471,
  "Delivery_location_latitude": 22.765049,
  "Delivery_location_longitude": 75.912471,
  "Order_Date": "19-03-2022",
  "Time_Orderd": "11:00",
  "Time_Order_picked": "11:15",
  "Weatherconditions": "conditions Sunny",
  "Road_traffic_density": "High ",
  "Vehicle_condition": 2,
  "Type_of_order": "Snack ",
  "Type_of_vehicle": "motorcycle ",
  "multiple_deliveries": "0",
  "Festival": "No ",
  "City": "Urban "
}
```

**Response:**

```json
19.12
```

### `GET /docs`

Auto-generated Swagger UI for interactive API testing.

---

## 📓 Experimental Notebooks

The `notebooks/` directory contains the research journey that led to the final model:

| Notebook | Purpose | Key Finding |
| :--- | :--- | :--- |
| `EDA.ipynb` | Exploratory Data Analysis | Identified skewed target distribution, coordinate anomalies, and key feature correlations |
| `Exp1dropVSimpute.ipynb` | Drop missing values vs imputation | Dropping performed better than imputation for this dataset's missing pattern |
| `Exp2missingIndicator.ipynb` | Missing indicator feature engineering | Adding missing indicators did not improve model performance |
| `Exp3modelSelectionfeatureeng.ipynb` | Model selection & feature engineering | Stacking (RF + XGBoost) outperformed individual models and LightGBM |
| `final_model_building.ipynb` | Final model with `TransformedTargetRegressor` | PowerTransformer on target + Stacking achieved the best generalization |

---

## 🛠 Tech Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **ML Framework** | Scikit-Learn, XGBoost |
| **Pipeline Orchestration** | DVC (Data Version Control) |
| **Experiment Tracking** | MLflow (hosted on DagsHub) |
| **Model Registry** | MLflow Model Registry |
| **API Framework** | FastAPI + Uvicorn |
| **Frontend** | Jinja2 + HTML/CSS/JS |
| **Data Validation** | Pydantic |
| **Serialization** | Joblib |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Aryan Prajapati](https://github.com/aryann13)**

</div>
