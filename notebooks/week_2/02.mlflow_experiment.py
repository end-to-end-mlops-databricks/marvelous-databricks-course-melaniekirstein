# Databricks notebook source
# MAGIC  %md
# MAGIC # MLflow Experiment for Hotel Reservations
# MAGIC **Notebook**: 02.mlflow_experiment.py
# MAGIC
# MAGIC ## Overview:
# MAGIC 1. Install necessary packages and restart the Python environment.
# MAGIC 2. Set up MLflow tracking and experiment configurations.
# MAGIC 3. Search for existing experiments related to the hotel reservation project.
# MAGIC 4. Save experiment details to a JSON file.
# MAGIC 5. Log parameters and metrics for an MLflow run.
# MAGIC 6. Retrieve run details and save them to a JSON file.
# MAGIC 7. Print metrics and parameters from the run information.

# COMMAND ----------
# MAGIC #%pip install ../hotel_reservations-0.0.1-py3-none-any.whl
# Installs the necessary package for handling hotel reservation data processing.

# COMMAND ----------
# dbutils.library.restartPython()
# Restarts the Python environment to ensure package installations take effect.

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Set Up MLflow Tracking and Experiment Configurations
# MAGIC Set the MLflow tracking URI, experiment name, and tags for the hotel reservations project.

# COMMAND ----------
import json
import mlflow

# Set the MLflow tracking URI and experiment details
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-mk")
mlflow.set_experiment_tags({"repository_name": "hotel-reservations-mk"})

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Search for Existing Experiments
# MAGIC Search for existing MLflow experiments related to the "hotel-reservations-mk" project.

# COMMAND ----------
# Search for experiments related to the project
experiments = mlflow.search_experiments(filter_string="tags.repository_name='hotel-reservations-mk'")
print(experiments)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Save Experiment Details to a JSON File
# MAGIC Save the details of the first experiment to a JSON file for later reference.

# COMMAND ----------
# Save experiment details to a JSON file
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Log Parameters and Metrics for an MLflow Run
# MAGIC Start a new MLflow run, log parameters, and metrics for tracking the experiment.

# COMMAND ----------
# Start a new MLflow run and log parameters and metrics
with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": "71f8c100e9c90b43fb52c580468aa675c630454e", "branch": "week2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5. Retrieve Run Information and Save It to a JSON File
# MAGIC Fetch detailed run information using the run ID and save it to a JSON file.

# COMMAND ----------
# Retrieve the run information using the run ID
run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel-reservations-mk"],
    filter_string="tags.git_sha='71f8c100e9c90b43fb52c580468aa675c630454e'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------
# Save the run details to a JSON file
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6. Print Metrics and Parameters from Run Information
# MAGIC Print out the logged metrics and parameters for review.

# COMMAND ----------
# Print the metrics and parameters from the run information
print(run_info["data"]["metrics"])
print(run_info["data"]["params"])
