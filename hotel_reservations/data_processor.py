import pandas as pd
import yaml
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pyspark.sql import SparkSession
from hotel_reservations.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df
        self.config = config

    def preprocess_data(self):
        """
        Preprocess the data according to the configuration.

        This includes removing rows with missing target values, separating features
        and target variables, and setting up preprocessing pipelines for numeric
        and categorical features.

        Returns:
        tuple: (self.X, self.y, self.preprocessor)
            self.X (pandas.DataFrame): The feature dataframe.
            self.y (pandas.Series): The target series.
            self.preprocessor (ColumnTransformer): The preprocessing pipeline.
        """
        # Remove rows with missing values in the target column
        target = self.config.target
        self.df = self.df.dropna(subset=[target])

        id_field = self.config.id_field

        # Separate features and target variable based on configuration
        self.X = self.df[[id_field] + self.config.num_features + self.config.cat_features]
        self.y = self.df[target]

        # Create preprocessing steps for numeric data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        # Create preprocessing steps for categorical data
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine numeric and categorical preprocessing steps into a single transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.num_features),
                ("cat", categorical_transformer, self.config.cat_features),
            ]
        )

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Parameters:
        test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 42).

        Returns:
        tuple: (X_train, X_test, y_train, y_test)
            X_train (pandas.DataFrame): Training features.
            X_test (pandas.DataFrame): Testing features.
            y_train (pandas.Series): Training target variable.
            y_test (pandas.Series): Testing target variable.
        """
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
