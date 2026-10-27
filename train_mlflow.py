
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Configure local database for tracking and model registration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MLflow_Model_Comparison")

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "Logistic_Regression": LogisticRegression(
        max_iter=500, random_state=42
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
}

best_accuracy = -1
best_model_name = ""
best_run_id = ""

# Train and evaluate both models
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )

        # Log parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        if name == "Random_Forest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)
        else:
            mlflow.log_param("max_iter", 500)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log trained model as an artifact
        signature = infer_signature(
            X_train, model.predict(X_train)
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5]
        )

        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Select best model based on test accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_run_id = run.info.run_id

# Register the best model
model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="Best_Iris_Classifier"
)

print("\nBest Model:", best_model_name)
print(f"Best Accuracy: {best_accuracy:.4f}")
print("Best Run ID:", best_run_id)
print("Registered Model Version:", registered_model.version)
