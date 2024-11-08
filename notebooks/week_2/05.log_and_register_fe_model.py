# Databricks notebook source
# MAGIC %md
# MAGIC # Log and Register Feature Engineering Model for Hotel Reservations
# MAGIC **Notebook**: 05.log_and_register_fe_model.py
# MAGIC
# MAGIC ## Overview:
# MAGIC 1. Install necessary packages and restart the Python environment.
# MAGIC 2. Import required libraries and initialize the Databricks session.
# MAGIC 3. Load project configuration settings.
# MAGIC 4. Load and prepare training and testing datasets.
# MAGIC 5. Create and configure feature tables and functions.
# MAGIC 6. Prepare the data for feature engineering and model training.
# MAGIC 7. Set up and train a model with feature engineering.
# MAGIC 8. Log the model, parameters, metrics, and feature-engineered data in MLflow.
# MAGIC 9. Register the model in the MLflow Model Registry.

# COMMAND ----------
# MAGIC #%pip install ../hotel_reservations-0.0.1-py3-none-any.whl
# Install the required package for feature engineering and hotel reservations processing.

# COMMAND ----------
# dbutils.library.restartPython()
# Restart the Python environment to apply the package installations.

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Import Libraries and Initialize Databricks Session
# MAGIC Import necessary libraries and initialize the session with Databricks clients.

# COMMAND ----------
import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
from pyspark.sql import functions as F
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from hotel_reservations.config import ProjectConfig  # Adjust as necessary

# Initialize Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Load Configuration Settings
# MAGIC Load configuration details for features, target, and model parameters.

# COMMAND ----------
# Set the MLflow registry and tracking URIs
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Load project configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration settings for features and target
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names for feature storage and function definition
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
function_name = f"{catalog_name}.{schema_name}.calculate_loyalty_score"

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Load and Prepare Datasets
# MAGIC Load the training and test datasets, and create a new table for feature storage.

# COMMAND ----------
# Load training and test sets from Databricks tables
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Create Feature Tables and Functions
# MAGIC Create and configure tables for storing features and define a function for calculating loyalty score.

# COMMAND ----------
# Create or replace the hotel_features table in Databricks
spark.sql(f"""
    CREATE OR REPLACE TABLE {feature_table_name}(
        booking_id STRING NOT NULL,
        lead_time INT,
        no_of_special_requests INT,
        avg_price_per_room FLOAT);
    """)

# Add constraints to the table and enable change data feed
spark.sql(f"""ALTER TABLE {feature_table_name}
          ADD CONSTRAINT hotel_pk PRIMARY KEY(booking_id);""")
spark.sql(f"""ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true);""")

# Insert data into the feature table from both train and test sets
spark.sql(f"""
        INSERT INTO {feature_table_name}
        SELECT
            booking_id, lead_time, no_of_special_requests, avg_price_per_room
        FROM {catalog_name}.{schema_name}.train_set
        """)
spark.sql(f"""
        INSERT INTO {feature_table_name}
        SELECT
            booking_id, lead_time, no_of_special_requests, avg_price_per_room
        FROM {catalog_name}.{schema_name}.test_set""")

# Define a custom function to calculate loyalty score
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(
    no_of_previous_cancellations DOUBLE,
    no_of_previous_bookings_not_canceled DOUBLE
)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
    # Define weightings
    w1 = 1.5     # Weight the number of times a previous booking was NOT cancelled
    w2 = 1.0     # Weight the number of times a previous booking was cancelled

    # Calculate loyalty score
    loyalty_score = (w1 * no_of_previous_bookings_not_canceled) - (w2 * no_of_previous_cancellations)
    return loyalty_score
$$
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5. Prepare Data for Feature Engineering
# MAGIC Transform the datasets, apply feature engineering, and prepare the final training data.

# COMMAND ----------
# Clean and prepare the training and test sets
train_set = train_set.drop("lead_time", "no_of_special_requests", "avg_price_per_room")
test_set = test_set.toPandas()

# Cast columns to appropriate types for feature engineering
train_set = train_set.withColumn(
    "no_of_previous_bookings_not_canceled", train_set["no_of_previous_bookings_not_canceled"].cast("double")
)
train_set = train_set.withColumn(
    "no_of_previous_cancellations", train_set["no_of_previous_cancellations"].cast("double")
)
train_set = train_set.withColumn("booking_id", train_set["booking_id"].cast("string"))

# Create the feature engineering training set
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["lead_time", "no_of_special_requests", "avg_price_per_room"],
            lookup_key="booking_id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="loyalty_score",
            input_bindings={
                "no_of_previous_cancellations": "no_of_previous_cancellations",
                "no_of_previous_bookings_not_canceled": "no_of_previous_bookings_not_canceled",
            },
        ),
    ],
)

# Load the feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate loyalty score for the test set
test_set["loyalty_score"] = (test_set["no_of_previous_bookings_not_canceled"] * 1.5) + (
    test_set["no_of_week_nights"] * 1.0
)

# Split features and target for model training
X_train = training_df[num_features + cat_features + ["loyalty_score"]]
y_train = training_df[target]
X_test = test_set[num_features + cat_features + ["loyalty_score"]]  # Ensure test set has the same features
y_test = test_set[target]

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6. Setup Model Pipeline and Train
# MAGIC Configure the preprocessing pipeline, set up the model, and train it with the feature-engineered data.

# COMMAND ----------
# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])

# COMMAND ----------
# MAGIC %md
# MAGIC ### 7. Train and Log Model in MLflow
# MAGIC Train the model, log parameters, metrics, and the model with feature engineering in MLflow.

# COMMAND ----------
# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-mk-fe")
git_sha = "71f8c100e9c90b43fb52c580468aa675c630454e"

# Start an MLflow run for logging the model
with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)

    # Log model with feature engineering
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model-fe",
    name=f"{catalog_name}.{schema_name}.hotel_reservations_model_fe",
)

# COMMAND ----------
#
