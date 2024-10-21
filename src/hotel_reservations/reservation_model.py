from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class ReservationModel:
    def __init__(self, preprocessor, config):
        """Initialize the ReservationModel with a preprocessor and configuration."""
        self.config = config
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Preprocessing step
            ('classifier', RandomForestClassifier(
                n_estimators=config['parameters']['n_estimators'],
                random_state=config['parameters'].get('random_state', 42)  # Random state for reproducibility
            ))
        ])

    def train(self, X_train, y_train):
        """
        Train the model using the training data.

        Parameters:
        X_train (pandas.DataFrame): The training features.
        y_train (pandas.Series): The target variable for training.
        """
        self.model.fit(X_train, y_train)  # Fit the model

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        X (pandas.DataFrame): The input features for prediction.

        Returns:
        numpy.ndarray: Predicted labels for the input features.
        """
        return self.model.predict(X)  # Return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using test data and return accuracy and classification report.

        Parameters:
        X_test (pandas.DataFrame): The test features.
        y_test (pandas.Series): The true labels for the test set.

        Returns:
        tuple: (accuracy (float), class_report (str))
            accuracy: Accuracy score of the model.
            class_report: Classification report as a string.
        """
        # Encode the test set target variable
        # y_test_encoded = self.label_encoder.transform(y_test)  # Optional: Encode target variable

        y_pred = self.predict(X_test)  # Predict on test set
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        class_report = classification_report(y_test, y_pred)  # Generate classification report
        return accuracy, class_report  # Return results

    def get_feature_importance(self):
        """
        Retrieve feature importances from the trained model.

        Returns:
        tuple: (feature_importance (numpy.ndarray), feature_names (numpy.ndarray))
            feature_importance: Array of feature importances.
            feature_names: Names of the features used in the model.
        """
        feature_importance = self.model.named_steps['classifier'].feature_importances_  # Get feature importances
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()  # Get feature names
        return feature_importance, feature_names  # Return importances and names

    def get_confusion_matrix(self, y_test, y_pred):
        """
        Generate and return confusion matrix and display.

        Parameters:
        y_test (pandas.Series): The true labels for the test set.
        y_pred (numpy.ndarray): The predicted labels from the model.

        Returns:
        tuple: (disp (ConfusionMatrixDisplay), cm (numpy.ndarray))
            disp: ConfusionMatrixDisplay object for visualizing the confusion matrix.
            cm: Confusion matrix as a 2D array.
        """
        cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)  # Prepare display
        return disp, cm  # Return display and confusion matrix
