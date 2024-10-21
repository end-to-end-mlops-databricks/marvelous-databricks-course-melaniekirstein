import yaml
import logging
import numpy as np

from src.hotel_reservations.data_processor import DataProcessor
from src.hotel_reservations.reservation_model import ReservationModel
from src.hotel_reservations.utils import visualize_results, plot_feature_importance, plot_confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from YAML file
with open('project_config.yml', 'r') as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# Initialize DataProcessor
data_processor = DataProcessor('data/Hotel Reservations.csv', config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess_data()
logger.info("Data preprocessed.")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = data_processor.split_data()
logger.info("Data split into training and test sets.")
logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Initialize and train the model
model = ReservationModel(data_processor.preprocessor, config)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
model.train(X_train, y_train)
logger.info("Model training completed.")

# Evaluate the model
accuracy, class_report = model.evaluate(X_test, y_test)
logger.info(f"Model evaluation completed: Accuracy={accuracy}")
logger.info(f"\n{class_report}")

# Visualizing Results
y_pred = model.predict(X_test)
visualize_results(y_test, y_pred)
logger.info("Results visualization completed.")

# Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)
logger.info("Feature importance plot generated.")

# Confusion Matrix
disp, cm = model.get_confusion_matrix(y_test, y_pred)
plot_confusion_matrix(disp, cm)
logger.info("Confusion Matrix plot generated.")
