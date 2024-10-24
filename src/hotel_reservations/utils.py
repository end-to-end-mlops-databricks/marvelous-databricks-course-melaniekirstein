import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_results(y_test, y_pred):
    """
    Visualize the actual vs predicted booking status using a count plot.

    Parameters:
    y_test (pandas.Series): The true booking statuses.
    y_pred (numpy.ndarray): The predicted booking statuses.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_test, hue=y_pred, palette="pastel")  # Count plot of actual vs predicted
    plt.title("Actual vs Predicted Booking Status")  # Title of the plot
    plt.xlabel("Actual Status")  # X-axis label
    plt.ylabel("Predicted Status")  # Y-axis label
    plt.legend(title="Predicted", loc="upper right")  # Legend settings
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()  # Display the plot


def plot_feature_importance(feature_importance, feature_names, top_n=10):
    """
    Plot the top N feature importances in a horizontal bar chart.

    Parameters:
    feature_importance (numpy.ndarray): The importance scores of features.
    feature_names (numpy.ndarray): The names of the features.
    top_n (int): Number of top features to display. Default is 10.
    """
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)  # Sort indices of feature importance
    pos = np.arange(sorted_idx[-top_n:].shape[0]) + 0.5  # Position for bars
    plt.barh(pos, feature_importance[sorted_idx[-top_n:]])  # Horizontal bar chart
    plt.yticks(pos, feature_names[sorted_idx[-top_n:]])  # Set y-tick labels
    plt.title(f"Top {top_n} Feature Importance")  # Title of the plot
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()  # Display the plot


def plot_confusion_matrix(disp, cm):
    """
    Plot the confusion matrix.

    Parameters:
    disp (ConfusionMatrixDisplay): The ConfusionMatrixDisplay object to visualize the matrix.
    cm (numpy.ndarray): The confusion matrix as a 2D array.
    """
    disp.plot(cmap=plt.cm.Blues)  # Plot confusion matrix with a color map
    plt.title("Confusion Matrix")  # Title of the plot
    plt.show()  # Display the plot
