import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# Example regressors
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load your dataset
# X, y = ... (Load your features and target variables here)
# For demonstration, let's create a dummy dataset:
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and parameters
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=3),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100)
}

# Start MLflow experiment
experiment_name = "Regression Models Experiment"
mlflow.set_experiment(experiment_name)

# Train and log each model
for model_name, model in models.items():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log model parameters and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("timestamp", timestamp)
        if hasattr(model, 'get_params'):
            for param_name, param_value in model.get_params().items():
                mlflow.log_param(param_name, param_value)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log the model
        if model_name == "XGBoostRegressor":
            mlflow.xgboost.log_model(model, model_name)
        elif model_name == "LightGBMRegressor":
            mlflow.lightgbm.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

print("Training completed and logged to MLflow.")

# Retrieve the latest run for a given experiment
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                          order_by=["start_time DESC"], max_results=1)
latest_run = runs[0]
print(f"Latest run ID: {latest_run.info.run_id}")
print(f"Run name: {latest_run.data.tags['mlflow.runName']}")
print(f"Run start time: {latest_run.info.start_time}")

# Access the run folder directly if needed
run_id = latest_run.info.run_id
run_folder_path = f"mlruns/{experiment.experiment_id}/{run_id}"
print(f"Run folder path: {run_folder_path}")
