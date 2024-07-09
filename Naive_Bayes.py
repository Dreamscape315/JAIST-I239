import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from utils import plot_confusion_matrix, display_evaluation_metrics

datasets = {
    "Iris": load_iris(),
    "Wine": load_wine()
}


def Naive_Bayes(data, dataset_name, confusion_matrix=False, evaluation_metrics=False, evaluation_metrics_return=False):
    """
    Naive Bayes classifier.
    Parameters:
    - data: dataset
    - dataset_name: str Name of the dataset.
    - confusion_matrix: bool If True, plots the confusion matrix. Default is False.
    - evaluation_metrics: bool If True, displays the evaluation metrics. Default is False.
    - evaluation_metrics_return: bool If True, returns the evaluation metrics as a tuple. Default is False.
    """

    X = data.data[:, 1:3]
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if confusion_matrix:
        plot_confusion_matrix(y_test, y_pred, data.target_names, dataset_name, 'Naive Bayes')
    if evaluation_metrics:
        if evaluation_metrics_return:
            return display_evaluation_metrics(y_test, y_pred, dataset_name, 'Naive Bayes', evaluation_metrics_return)
        else:
            display_evaluation_metrics(y_test, y_pred, dataset_name, 'Naive Bayes', evaluation_metrics_return)


for dataset_name, data in datasets.items():
    Naive_Bayes(data, dataset_name, confusion_matrix=True, evaluation_metrics=True, evaluation_metrics_return=False)
