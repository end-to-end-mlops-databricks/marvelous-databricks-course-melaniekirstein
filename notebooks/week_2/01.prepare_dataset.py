# Databricks notebook source
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.config import ProjectConfig
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)

# COMMAND ----------
# Load the hotel reservation dataset
df = spark.read.csv(
    "/Volumes/heiaepgah71pwedmld01001/mk_test/mlops_data/Hotel Reservations.csv",
    header=True,
    inferSchema=True).toPandas()

# COMMAND ----------
data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess_data()
X_train, X_test, y_train, y_test = data_processor.split_data()
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
