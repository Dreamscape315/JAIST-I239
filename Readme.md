# JAIST-I239
## Machine Learning 1-2 2024
This is the report for Machine Learning i239 course in JAIST.

In this report, we will discuss the following fundamental machine learning algorithms:
1. SVM Support Vector Machine
2. Decision Tree
3. Naive Bayes

## Datasets
- Iris (only choose 2 features)
- Wine (only choose 2 features)

## Functions
- `SVM` algorithm
- `Decision_Tree` algorithm
- `Naive_Bayes` algorithm
- `plot_confusion_matrix` calculate and plot confusion matrix
- `display_evaluation_metrics` display evaluation metrics or return it
- `Compare` compare the performance of different algorithms

### SVM Support Vector Machine

| Datasets  | Accuracy | Precision | Recall | F1-score |
|:----------|:--------:|----------:|-------:|---------:|
| Iris      |  0.9333  |    0.9436 | 0.9333 |   0.9319 |
| Wine      |  0.5833  |    0.5880 | 0.5833 |   0.5829 |

The boundary of the iris by SVM algorithm is shown in the following figure:

![svm_boundary_iris.png](Results%2Fsvm_boundary_iris.png)

The boundary of the wine by SVM algorithm is shown in the following figure:

![svm_boundary_wine.png](Results%2Fsvm_boundary_wine.png)

The confusion matrix is shown in the `Results` folder.

`Results/svm_matrix_iris.png`

`Results/svm_matrix_wine.png`

### Decision Tree

| Datasets  | Accuracy | Precision | Recall | F1-score |
|:----------|:--------:|----------:|-------:|---------:|
| Iris      |  0.8333  |    0.8367 | 0.8333 |   0.8338 |
| Wine      |  0.6111  |    0.6310 | 0.6111 |   0.6159 |

The decision tree of the iris is shown in the following figure:

[Decision Tree Visualization of Iris by Decision Tree.pdf](Results%2FDecision%20Tree%20Visualization%20of%20Iris%20by%20Decision%20Tree.pdf)

The decision tree of the wine is shown in the following figure:

[Decision Tree Visualization of Wine by Decision Tree.pdf](Results%2FDecision%20Tree%20Visualization%20of%20Wine%20by%20Decision%20Tree.pdf)

The confusion matrix is shown in the `Results` folder.

`Results/dt_matrix_iris.png`

`Results/dt_matrix_wine.png`


### Naive Bayes

| Datasets  | Accuracy | Precision | Recall | F1-score |
|:----------|:--------:|----------:|-------:|---------:|
| Iris      |  1.0000  |    1.0000 | 1.0000 |   1.0000 |
| Wine      |  0.6389  |    0.6892 | 0.6389 |   0.6451 |

The confusion matrix is shown in the `Results` folder.

`Results/nb_matrix_iris.png`

`Results/nb_matrix_wine.png`

## Conclusion

### Performance Comparison

| Datasets | Model | Accuracy | Precision | Recall | F1-score |
|:--------:|:-----:|:--------:|:---------:|:------:|:--------:|
| Iris     | SVM   | 0.9333   |  0.9436   | 0.9333 |  0.9319  |
|          | DT    | 0.8333   |  0.8367   | 0.8333 |  0.8338  |
|          | NB    | 1.0000   |  1.0000   | 1.0000 |  1.0000  |
| Wine     | SVM   | 0.5833   |  0.5880   | 0.5833 |  0.5829  |
|          | DT    | 0.6111   |  0.6173   | 0.6111 |  0.6121  |
|          | NB    | 0.6389   |  0.6892   | 0.6389 |  0.6451  |


### Summary
#### Model Performance Comparison
- In the Iris dataset, the Naive Bayes algorithm has the best performance.
- In comparison, the SVM model performs slightly lower on the Iris dataset but is still quite good.
- The Decision Tree model has the lowest performance on the Iris dataset.

#### Dataset Difficulty Comparison

- For the Wine dataset, the performance of all models is significantly lower than on the Iris dataset, which may indicate that the Wine dataset is more difficult to classify. 

#### Model Generality

- The Naive Bayes model shows a large difference in performance on the two datasets, which may indicate its limited adaptability to datasets with different characteristics.
- The SVM and Decision Tree models have more similar performances on the two datasets, indicating that these two models may have better generality.
- But Naive Bayes still has the best performance than the other two models.
