import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import plot_confusion_matrix, display_evaluation_metrics

datasets = {
    "Iris": load_iris(),
    "Wine": load_wine()
}


def SVM(data, dataset_name, decision_boundary=False, confusion_matrix=False, evaluation_metrics=False,
        evaluation_metrics_return=False):
    """
    Support Vector Machine (SVM) classifier with linear kernel.
    Parameters:
    - data: dataset
    - dataset_name: str Name of the dataset.
    - decision_boundary: bool If True, plots the decision boundary. Default is False.
    - confusion_matrix: bool If True, plots the confusion matrix. Default is False.
    - evaluation_metrics: bool If True, displays the evaluation metrics. Default is False.
    - evaluation_metrics_return: bool If True, returns the evaluation metrics as a tuple. Default is False.
    """
    X = data.data[:, 1:3]
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = svm.SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if decision_boundary:
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # Predict class labels for each point in the mesh grid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
        plt.xlabel(data.feature_names[1])
        plt.ylabel(data.feature_names[2])
        plt.title('SVM Decision Boundary with 2 Features')
        plt.show()
    if confusion_matrix:
        plot_confusion_matrix(y_test, y_pred, data.target_names, dataset_name, 'SVM')
    if evaluation_metrics:
        if evaluation_metrics_return:
            return display_evaluation_metrics(y_test, y_pred, dataset_name, 'SVM', evaluation_metrics_return)
        else:
            display_evaluation_metrics(y_test, y_pred, dataset_name, 'SVM', evaluation_metrics_return)


for dataset_name, data in datasets.items():
    SVM(data, dataset_name, decision_boundary=True, confusion_matrix=True, evaluation_metrics=True, evaluation_metrics_return=False)
