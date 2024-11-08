# Databricks notebook source
# MAGIC  %md
# MAGIC # Preparing Dataset
# MAGIC **Notebook**: 01.prepare_dataset.py
# MAGIC
# MAGIC ## Overview:
# MAGIC 1. Install necessary packages and restart the Python environment.
# MAGIC 2. Import required libraries and initialize the Spark session.
# MAGIC 3. Load project configuration settings.
# MAGIC 4. Load the hotel reservation dataset.
# MAGIC 5. Preprocess the data, split it into training and testing sets, and save it to the data catalog.

# COMMAND ----------
# MAGIC #%pip install ../hotel_reservations-0.0.1-py3-none-any.whl
# Installs the necessary package for handling hotel reservation data processing.

# COMMAND ----------
# dbutils.library.restartPython()
# Restarts the Python environment to ensure package installations take effect.

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Import Libraries and Initialize Spark Session
# MAGIC Import essential libraries and initialize a Spark session for distributed data processing.

# COMMAND ----------
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.config import ProjectConfig
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession

# Create or retrieve an existing Spark session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Load Configuration
# MAGIC Load project configuration settings for the project from a YAML file.

# COMMAND ----------
# Load project configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Load Dataset
# MAGIC Load the hotel reservation dataset from a specified path, inferring schema and including headers.

# COMMAND ----------
# Load the hotel reservation dataset
df = spark.read.csv(
    "/Volumes/heiaepgah71pwedmld01001/hotel_reservations_mk/raw_data/Hotel Reservations.csv",
    header=True,  # Specify that the CSV file contains header row
    inferSchema=True,  # Enable schema inference
).toPandas()  # Convert Spark DataFrame to Pandas DataFrame for local processing

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Preprocess, Split, and Save Data
# MAGIC Preprocess the data using the DataProcessor, split it into training and testing sets, and save it to the catalog.

# COMMAND ----------
# Initialize data processor with configuration and dataset
data_processor = DataProcessor(pandas_df=df, config=config)

# Preprocess the dataset (e.g., handle missing values, normalize features)
data_processor.preprocess_data()

# Split data into features (X) and target (y) for training and testing
X_train, X_test, y_train, y_test = data_processor.split_data()

# Combine features and target for training and testing sets
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# Save the prepared datasets to the data catalog
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
