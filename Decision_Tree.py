from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from utils import plot_confusion_matrix, display_evaluation_metrics

datasets = {
    "Iris": load_iris(),
    "Wine": load_wine()
}


def Decision_Tree(data, dataset_name, decision_tree=False, confusion_matrix=False, evaluation_metrics=False,
                  evaluation_metrics_return=False):
    """
    Decision Tree classifier with entropy criterion.
    Parameters:
    - data: dataset
    - dataset_name: str Name of the dataset.
    - decision_tree: bool If True, plots the decision tree. Default is False.
    - confusion_matrix: bool If True, plots the confusion matrix. Default is False.
    - evaluation_metrics: bool If True, displays the evaluation metrics. Default is False.
    - evaluation_metrics_return: bool If True, returns the evaluation metrics as a tuple. Default is False.
    """

    X = data.data[:, 1:3]
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if decision_tree:
        # Plot the decision tree
        plt.figure(figsize=(20, 20))
        plot_tree(model, feature_names=data.feature_names[1:3], class_names=data.target_names, filled=True)
        plt.savefig(f"Decision Tree Visualization of {dataset_name} by Decision Tree.pdf")
        plt.title(f"Decision Tree Visualization of {dataset_name} by Decision Tree")
        plt.show()
    if confusion_matrix:
        plot_confusion_matrix(y_test, y_pred, data.target_names, dataset_name, 'Decision Tree')
    if evaluation_metrics:
        if evaluation_metrics_return:
            return display_evaluation_metrics(y_test, y_pred, dataset_name, 'Decision Tree', evaluation_metrics_return)
        else:
            display_evaluation_metrics(y_test, y_pred, dataset_name, 'Decision Tree', evaluation_metrics_return)


for dataset_name, data in datasets.items():
    Decision_Tree(data, dataset_name, decision_tree=True, confusion_matrix=True, evaluation_metrics=True, evaluation_metrics_return=False)