import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

with mlflow.start_run() as run:
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=900,
        random_state=4
    )
    model.fit(X, y)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 900)
    mlflow.log_metric("train_acc", model.score(X, y))

    # Log the model with the trusted scikit-learn tree type
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # Display results
    print("Run ID:", run.info.run_id)
    print("Training accuracy:", model.score(X, y))
    print("Model logged successfully!")

