# Databricks notebook source

# MAGIC #%pip install ../hotel_reservations-0.0.1-py3-none-any.whl

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Online Table for House Features
# MAGIC This section demonstrates the creation of an online table to store the house features used in the model. 
# MAGIC The `hotel_features` table has been created as a feature lookup table.

# COMMAND ----------

import time

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Load Configuration
# MAGIC This loads the configuration details, such as catalog and schema names, from the project configuration file.

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Create Online Table
# MAGIC We create an online table for `hotel_reservation_features_online` using the source `hotel_features`. 
# MAGIC The table will have a primary key of `booking_id`.

# COMMAND ----------

online_table_name = f"{catalog_name}.{schema_name}.hotel_reservation_features_online"
spec = OnlineTableSpec(
    primary_key_columns=["booking_id"],
    source_table_full_name=f"{catalog_name}.{schema_name}.hotel_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create Endpoint for Model Serving with Feature Lookup
# MAGIC This section demonstrates creating an endpoint to serve the model, incorporating the feature lookup table.

# COMMAND ----------

workspace.serving_endpoints.create(
    name="hotel-reservations-mk-model-serving-fe",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.hotel_reservations_model_fe",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Call the Endpoint
# MAGIC This section demonstrates how to call the model endpoint to make predictions, including the use of the feature lookup table.

# COMMAND ----------

dbutils = DBUtils(spark)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Prepare Sample Request Data
# MAGIC Here, we sample records from the training set and create the request body to call the model endpoint. 
# MAGIC Certain columns like "OverallQual", "GrLivArea", and "GarageCars" will be retrieved from the feature lookup.

# COMMAND ----------

# Excluding "OverallQual", "GrLivArea", "GarageCars" because they will be taken from feature lookup
required_columns = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    # "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    # "avg_price_per_room",
    # "no_of_special_requests",
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
    "booking_id",
]

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

train_set.dtypes

# COMMAND ----------

dataframe_records[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Call the Model Endpoint for Prediction
# MAGIC In this step, we make a request to the model serving endpoint with the sampled data.

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-mk-model-serving-fe/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Load the Feature Lookup Table
# MAGIC We load the `hotel_features` table, which is used for feature lookup.

# COMMAND ----------

house_features = spark.table(f"{catalog_name}.{schema_name}.hotel_features").toPandas()

# COMMAND ----------

house_features.dtypes
