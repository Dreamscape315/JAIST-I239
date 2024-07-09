from sklearn.datasets import load_iris, load_wine
import pandas as pd
from Decision_Tree import Decision_Tree
from SVM import SVM
from Naive_Bayes import Naive_Bayes

datasets = {
    "Iris": load_iris(),
    "Wine": load_wine()
}

result = pd.DataFrame(columns=['Dataset', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


def Compare():
    for dataset_name, data in datasets.items():
        svm = SVM(data, dataset_name, decision_boundary=False, confusion_matrix=False, evaluation_metrics=True,
                  evaluation_metrics_return=True)
        dt = Decision_Tree(data, dataset_name, decision_tree=False, confusion_matrix=False, evaluation_metrics=True,
                           evaluation_metrics_return=True)
        nb = Naive_Bayes(data, dataset_name, confusion_matrix=False, evaluation_metrics=True,
                         evaluation_metrics_return=True)
        result.loc[len(result)] = [dataset_name, 'SVM', svm[0], svm[1], svm[2], svm[3]]
        result.loc[len(result)] = [dataset_name, 'Decision Tree', dt[0], dt[1], dt[2], dt[3]]
        result.loc[len(result)] = [dataset_name, 'Naive Bayes', nb[0], nb[1], nb[2], nb[3]]
    print(result)


Compare()