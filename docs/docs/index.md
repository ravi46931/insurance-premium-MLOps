<!-- # Introduction
# Dataset
# Model Training
# Model Tracking
# Model Evaluation
# Model deployment -->

# Insurance Premium Prediction Application

## Overview

This is a machine learning application designed for predicting insurance premiums. The project leverages a variety of tools and frameworks to streamline data management, experiment tracking, and model deployment.

## Tools Utilized

- **DVC (Data Version Control)**: Used for managing and versioning data pipeline.
- **Git**: Version control system for tracking code changes.
- **MLflow**: Used for tracking the model training and model evaluation.
- **GitHub Actions Server**: Used for continuous integration and deployment.
- **Dagshub**: Facilitates MLflow experiment tracking and DVC data pipeline.

## Machine Learning Pipeline

### Data Ingestion

The application ingests insurance premium data from the specified original data path and saves it into `artifacts/DataIngestionArtifacts`.

### Data Transformation

Data undergoes transformation to prepare it for model training. Transformed data and preprocessing artifacts are saved into `artifacts/DataTransformationArtifacts`. Preprocessors are also stored in `models/`.

### Model Training

Multiple machine learning models are trained:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Catboost

The top 4 performing models based on training metrics are selected. Both models and associated metrics are saved into `artifacts/ModelTrainerArtifacts`. MLflow is used to track model parameters and metrics throughout this process.

### Model Evaluation

The best-performing model on test data is selected and saved into `artifacts/ModelEvaluationArtifacts` and `models/`. Model evaluation metrics are tracked using MLflow.

### Streamlit App Deployment

A Streamlit application is developed to allow users to input data and receive predictions from the selected model.

## Usage

To reproduce the model and run the application:

1. Clone the repository:
    
    `git clone <repository_url>`<br>
    `cd <repository_name>`
    
2. Set up the environment:

    `pip install -r requirements.txt`<br>

3. Run the Streamlit app:

    `streamlit run app.py`

## Directory Structure

```bash
.github/
└── workflows/
├── .gitkeep
└── ci.yaml
src/
├── init.py
├── components/
│ ├── init.py
│ ├── data_ingestion.py
│ ├── data_transformation.py
│ ├── model_trainer.py
│ └── model_evaluation.py
├── constants/
│ └── init.py
├── entity/
│ ├── init.py
│ ├── config_entity.py
│ └── artifact_entity.py
├── pipeline/
│ ├── init.py
│ ├── training_pipeline.py
│ └── prediction_pipeline.py
├── utils/
│ ├── init.py
│ └── utils.py
├── logger/
│ └── init.py
└── exception/
└── init.py
tests/
├── unit/
│ └── init.py
└── integration/
└── init.py
docs/
├── docs/
│ ├── index.md
│ └── getting-started.md
├── mkdocs.yml
└── README.md
data/
└── insurance.csv
.env
requirements.txt
setup.py
setup.cfg
pyproject.toml
tox.ini
app.py
main.py
experiment/
└── experiments.ipynb
README.md
implement.md
.gitignore
```

## Additional Tools
- Data Version Control (DVC): Tracks changes in datasets for reproducibility.
- MLflow with Dagshub: Manages experiments, parameters, and metrics across the ML lifecycle.

