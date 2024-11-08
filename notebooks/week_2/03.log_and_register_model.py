# Databricks notebook source
# MAGIC  %md
# MAGIC # Log and Register Model for Hotel Reservations
# MAGIC **Notebook**: 03.log_and_register_model.py
# MAGIC
# MAGIC ## Overview:
# MAGIC 1. Install necessary packages and restart the Python environment.
# MAGIC 2. Import required libraries and initialize MLflow.
# MAGIC 3. Load project configuration settings.
# MAGIC 4. Load training and testing datasets.
# MAGIC 5. Define preprocessing steps and create a machine learning pipeline.
# MAGIC 6. Train the model, log parameters, metrics, and the model to MLflow.
# MAGIC 7. Register the trained model to the MLflow Model Registry.
# MAGIC 8. Load the dataset used for training from the MLflow data context.

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
from pyspark.sql import SparkSession
from hotel_reservations.config import ProjectConfig
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
from mlflow.models import infer_signature

# Set the MLflow tracking URI and the registry URI for Unity Catalog
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog

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

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Load Training and Testing Datasets
# MAGIC Load the training and testing datasets from Databricks tables for model training.

# COMMAND ----------
# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Separate features and target variables for training and testing
X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Define Preprocessing and Create Pipeline
# MAGIC Define preprocessing steps and set up a pipeline with the LightGBM classifier.

# COMMAND ----------
# Define the preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

# Create the pipeline with preprocessing and the LightGBM classifier
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5. Train the Model and Log Information
# MAGIC Train the model, log parameters, metrics, and the model itself to MLflow.

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-mk")
git_sha = "71f8c100e9c90b43fb52c580468aa675c630454e"

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    run_id = run.info.run_id

    # Train the model and make predictions
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance using classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Print accuracy and classification report
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)

    # Log classification report metrics
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"precision_{class_label}", metrics["precision"])
            mlflow.log_metric(f"recall_{class_label}", metrics["recall"])
            mlflow.log_metric(f"f1-score_{class_label}", metrics["f1-score"])

    # Infer signature for model input and output
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log the training dataset as input to the model
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    # Log the trained model
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6. Register the Model in MLflow Model Registry
# MAGIC Register the trained model to the MLflow Model Registry for versioning and management.

# COMMAND ----------
# Register the trained model in the MLflow Model Registry
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=f"{catalog_name}.{schema_name}.hotel_reservations_model_basic",
    tags={"git_sha": f"{git_sha}"},
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 7. Load and Review Dataset from MLflow Data Context
# MAGIC Retrieve and load the dataset used for training from MLflow's data context.

# COMMAND ----------
# Get the dataset information from the MLflow run
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()
