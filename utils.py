import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name, model_name):
    """
    Plots the confusion matrix using seaborn heatmap.

    Parameters:
    - y_true: array-like of shape (n_samples,) Ground truth (correct) target values.
    - y_pred: array-like of shape (n_samples,) Estimated targets as returned by a classifier.
    - class_names: array-like of shape (n_classes,) List of class names for the target variable.
    - dataset_name: str Name of the dataset.
    - model_name: str Name of the model used for classification.
    """
    cm = confusion_matrix(y_true, y_pred)  # Compute the confusion matrix
    plt.figure(figsize=(10, 7))  # Set the figure size
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix of {dataset_name} by {model_name}')    # Set the title
    plt.show()


def display_evaluation_metrics(y_true, y_pred, dataset_name, model_name, return_metrics=False):
    """
    Displays the evaluation metrics including accuracy, precision, recall, and F1 score.

    Parameters:
    - y_true: array-like of shape (n_samples,) Ground truth (correct) target values.
    - y_pred: array-like of shape (n_samples,) Estimated targets as returned by a classifier.
    - dataset_name: str Name of the dataset.
    - model_name: str Name of the model used for classification.
    - return_metrics: bool If True, returns the evaluation metrics as a tuple. Default is False.
    """
    accuracy = accuracy_score(y_true, y_pred)   # Compute the accuracy
    precision = precision_score(y_true, y_pred, average='weighted')     # Compute the precision
    recall = recall_score(y_true, y_pred, average='weighted')   # Compute the recall
    f1 = f1_score(y_true, y_pred, average='weighted')   # Compute the F1 score

    if return_metrics:
        # Return the evaluation metrics if needed
        return f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"
    else:
        # Display the evaluation metrics
        print(f"------ Evaluation Metrics of {dataset_name} by {model_name}-----")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
