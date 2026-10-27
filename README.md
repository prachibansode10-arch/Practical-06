# MLOps Practical 6 — MLflow

## Aim
To install and configure MLflow, train machine learning models, log parameters and metrics, track artifacts, compare experiments, and register the best model.

## Software Requirements
- Python 3.10 or compatible version
- MLflow
- Scikit-learn
- Pandas
- Matplotlib

## Dataset
Iris flower dataset provided by Scikit-learn.

## Models Used
1. Logistic Regression
2. Random Forest Classifier

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## Procedure
1. Install the required Python packages.
2. Create and activate a virtual environment.
3. Train both classification models.
4. Log model parameters and evaluation metrics using MLflow.
5. Track trained models as artifacts.
6. Compare model runs using the MLflow Tracking UI.
7. Register the best model as `Best_Iris_Classifier`.

## Run the Project

```bash
python -m pip install -r requirements.txt
python train_mlflow.py
```

## Start MLflow UI

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in your browser.

## Result
The experiment compares Logistic Regression and Random Forest using evaluation metrics recorded by MLflow and registers the best model based on test accuracy.****
