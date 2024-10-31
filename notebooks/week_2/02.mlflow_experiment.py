# Databricks notebook source
import json

import mlflow

mlflow.set_tracking_uri("databricks")

mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-mk")
mlflow.set_experiment_tags({"repository_name": "hotel-reservations-mk"})

# COMMAND ----------
experiments = mlflow.search_experiments(filter_string="tags.repository_name='hotel-reservations-mk'")
print(experiments)

# COMMAND ----------
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)
# COMMAND ----------
with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": "50a9297454e49cbec3c6b681981b38f1485b3c10", "branch": "week2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})
# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel-reservations-mk"],
    filter_string="tags.git_sha='50a9297454e49cbec3c6b681981b38f1485b3c10'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------
print(run_info["data"]["metrics"])

# COMMAND ----------
print(run_info["data"]["params"])


# COMMAND ----------