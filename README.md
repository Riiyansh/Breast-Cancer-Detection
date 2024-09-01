
# Breast Cancer Detection Using Logistic Regression

This repository contains a Python implementation of a machine learning model for detecting breast cancer using Logistic Regression. The model is trained and evaluated using the `breast_cancer.csv` dataset.

## Project Overview

Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for successful treatment and improving survival rates. This project aims to build a predictive model that can classify whether a breast tumor is benign or malignant based on certain features extracted from breast tissue.

## What is Logistic Regression?

Logistic Regression is a type of statistical model used for binary classification tasks. Unlike Linear Regression, which predicts a continuous output, Logistic Regression predicts the probability that a given input belongs to a particular class. It is particularly useful when the dependent variable is categorical, such as in this project, where the task is to predict whether a tumor is benign or malignant.

The model works by fitting a logistic function (also known as the sigmoid function) to the data, which maps the input features to a probability value between 0 and 1. This probability is then used to classify the input into one of the two classes.

## Dataset

The dataset used in this project is the `breast_cancer.csv` file. It contains multiple features of breast cancer tumors and their corresponding labels (1 for malignant, 0 for benign). The dataset is split into two parts: features (`X`) and labels (`y`). The features are various measurements and characteristics of the tumors, while the labels indicate whether the tumor is malignant or benign.

## Steps Involved

### 1. Importing the Libraries

To start with, we need to import the necessary libraries. `pandas` is used for data manipulation and analysis.

```python
import pandas as pd
```

### 2. Importing the Dataset

The dataset is loaded into a pandas DataFrame using the `read_csv` function. After loading the data, we separate it into features (`X`) and the target variable (`y`).

```python
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

- `X` contains all the columns except the first (which might be an ID column) and the last one (which is the label).
- `y` contains the labels, indicating whether the tumor is malignant or benign.

### 3. Splitting the Dataset into Training and Test Sets

The dataset is then split into training and test sets. The training set is used to train the model, while the test set is used to evaluate its performance.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

- `test_size=0.2` indicates that 20% of the dataset is used for testing, and 80% is used for training.
- `random_state=0` ensures that the data split is reproducible.

### 4. Training the Logistic Regression Model

The Logistic Regression model is then trained using the training data. This involves fitting the model to the training features (`X_train`) and corresponding labels (`y_train`).

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
```

- The `fit` method finds the optimal parameters for the logistic function that best separates the two classes in the training data.

### 5. Predicting the Test Set Results

Once the model is trained, it is used to make predictions on the test set (`X_test`). The predicted labels (`y_pred`) are compared to the actual labels (`y_test`) to evaluate the model's performance.

```python
y_pred = classifier.predict(X_test)
```

- This step produces an array of predicted labels for the test set, which can then be compared with the actual labels to assess accuracy.

### 6. Making the Confusion Matrix

The confusion matrix is a tool used to evaluate the performance of the classification model by showing the number of correct and incorrect predictions for each class.

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Output: [[84  3] [ 3 47]]
```

- The matrix shows the number of true positives, true negatives, false positives, and false negatives. For example, `[84 3] [3 47]` indicates that there were 84 true positives, 47 true negatives, 3 false positives, and 3 false negatives.

### 7. Computing Accuracy with k-Fold Cross-Validation

To ensure the model's robustness and reliability, we use k-Fold Cross-Validation. This technique splits the training data into `k` subsets (folds) and trains the model `k` times, each time using a different fold as the validation set and the remaining `k-1` folds as the training set. The model's accuracy is averaged over the `k` iterations.

```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# Output: Accuracy: 96.70 %, Standard Deviation: 1.97 %
```

- `cv=10` indicates that we are using 10-fold cross-validation.
- The mean accuracy and standard deviation provide a measure of how well the model generalizes to unseen data.

## Results

The Logistic Regression model achieved an accuracy of **96.70%** with a standard deviation of **1.97%** using 10-fold cross-validation. The confusion matrix generated from the test set predictions shows a high level of accuracy, with only a few misclassifications.

## Conclusion

This project demonstrates the use of Logistic Regression for breast cancer detection. The model performs well, achieving high accuracy and low variance, making it a reliable tool for predicting breast cancer based on the provided dataset.

## How to Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/Riiyansh/breast-cancer-detection.git
   ```

2. Navigate to the project directory:
   ```
   cd breast-cancer-detection
   ```

3. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

4. Run the Python script:
   ```
   python breast_cancer_detection.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset was provided by [Source Name].
- Thanks to the open-source community for providing the tools and libraries used in this project.

---
