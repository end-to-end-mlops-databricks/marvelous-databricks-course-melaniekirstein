# Databricks notebook source
# MAGIC  %md
# MAGIC # Log and Register Custom Model for Hotel Reservations
# MAGIC **Notebook**: 04.log_and_register_custom_model.py
# MAGIC
# MAGIC ## Overview:
# MAGIC 1. Install necessary packages and restart the Python environment.
# MAGIC 2. Import required libraries and initialize MLflow.
# MAGIC 3. Load project configuration settings.
# MAGIC 4. Load the pre-trained model from MLflow.
# MAGIC 5. Define a custom wrapper for the model for predictions.
# MAGIC 6. Log custom model to MLflow, including the environment, signature, and input data.
# MAGIC 7. Register the custom model in the MLflow Model Registry.
# MAGIC 8. Load and use the registered model from the MLflow registry.

# COMMAND ----------
# MAGIC #%pip install ../hotel_reservations-0.0.1-py3-none-any.whl
# Installs the necessary package for handling hotel reservation data processing.

# COMMAND ----------
# dbutils.library.restartPython()
# Restarts the Python environment to ensure package installations take effect.

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Import Libraries and Initialize MLflow
# MAGIC Import necessary libraries, set up MLflow tracking URI, and prepare the environment for logging.

# COMMAND ----------
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from mlflow.models import infer_signature
from hotel_reservations.data_processor import ProjectConfig
import json
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env

# Set the MLflow tracking URI and the registry URI for Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Load Configuration Settings
# MAGIC Load configuration settings, including feature names and model parameters.

# COMMAND ----------
# Load project configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details for features and target
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Load Pre-trained Model from MLflow
# MAGIC Load a previously trained model from MLflow using a specific `run_id`.

# COMMAND ----------
# Fetch the run ID for the training process in the 'week2' branch
run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel-reservations-mk"],
    filter_string="tags.branch='week2'",
).run_id[0]

# Load the trained model from the run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Define Custom Model Wrapper
# MAGIC Define a custom wrapper for the model to support prediction and probability prediction.

# COMMAND ----------
# Define the custom model wrapper for predictions
class HotelReservationseModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input, return_proba=False):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Prediction based on specified mode
        if return_proba:
            # Predict probabilities for each class
            probabilities = self.model.predict_proba(model_input)
            predictions = {
                "Probabilities": probabilities.tolist(),
                "Predicted Class": probabilities.argmax(axis=1).tolist(),
            }
        else:
            # Predict class labels directly
            predicted_classes = self.model.predict(model_input)
            predictions = {"Predicted Class": predicted_classes.tolist()}

        return predictions

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5. Load Training and Testing Datasets
# MAGIC Load the datasets for training and testing from Databricks tables.

# COMMAND ----------
# Load the training and testing sets from Databricks
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6. Log the Custom Model to MLflow
# MAGIC Log the wrapped custom model along with input data, environment, and signature.

# COMMAND ----------
# Wrap the model with the custom wrapper class
wrapped_model = HotelReservationseModelWrapper(model)

# Select an example input for prediction
example_input = X_test.iloc[0:1]  # Use the first row as an example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# Set the MLflow experiment for logging
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-mk-pyfunc")
git_sha = "71f8c100e9c90b43fb52c580468aa675c630454e"

# Start an MLflow run for logging
with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})
    
    # Log the training dataset as input to the model
    dataset = mlflow.data.from_spark(train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    # Define Conda environment dependencies
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["code/hotel_reservations-0.0.1-py3-none-any.whl"],
        additional_conda_channels=None,
    )
    
    # Log the custom model with MLflow
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-hotel-reservations-mk-model",
        code_paths=["../hotel_reservations-0.0.1-py3-none-any.whl"],
        signature=signature,
    )

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-hotel-reservations-mk-model")
loaded_model.unwrap_python_model()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 7. Register the Custom Model in MLflow Model Registry
# MAGIC Register the custom model in the MLflow Model Registry to track and version it.

# COMMAND ----------
# Register the custom model in the MLflow Model Registry
model_name = f"{catalog_name}.{schema_name}.hotel_reservations_mk_pyfunc"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-hotel-reservations-mk-model", 
    name=model_name, 
    tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 8. Load and Use the Registered Model
# MAGIC Load the registered model from the Model Registry and use it for inference.

# COMMAND ----------
with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
# Load the model from the MLflow Model Registry using alias


model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
# Fetch model version by alias from the registry
client.get_model_version_by_alias(model_name, model_version_alias)
model
