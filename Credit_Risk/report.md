# Module 12 Report

## Overview of the Analysis

In this analysis, we aimed to build machine learning models to predict loan statuses based on financial information provided in the dataset. The dataset contained information on loan statuses labeled as `0` for healthy loans and `1` for high-risk loans. The goal was to develop models that could accurately classify loan statuses, particularly identifying high-risk loans.

The dataset comprised 75,036 healthy loans and 2,500 high-risk loans, indicating a class imbalance. To address this imbalance, we employed the RandomOverSampler technique from the imbalanced-learn library to resample the data and ensure equal representation of both classes in the training dataset.

We utilized logistic regression models to make predictions on both the original and resampled datasets. The performance of each model was evaluated based on metrics such as accuracy, precision, recall, and F1-score.

## Results

### Logistic Regression Model with Original Data:
- Balanced Accuracy Score: 0.9697
- Confusion Matrix:
  ```
  [[18658   107]
   [   34   585]]
  ```
- Classification Report:
  ```
                 precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.95      0.89       619

    accuracy                           0.99     19384
   macro avg       0.92      0.97      0.94     19384
weighted avg       0.99      0.99      0.99     19384
  ```

### Logistic Regression Model with Resampled Data:
- Balanced Accuracy Score: 0.9936
- Confusion Matrix:
  ```
  [[18646   119]
   [    4   615]]
  ```
- Classification Report:
  ```
                 precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384
  ```

## Summary

Both logistic regression models performed exceptionally well in predicting loan statuses, achieving high accuracy, precision, recall, and F1-scores. However, the model trained on the resampled data demonstrated slightly better performance, particularly in identifying high-risk loans. It achieved a higher balanced accuracy score and showed improved precision and recall for the minority class (`1` - high-risk loans).

Considering the importance of accurately identifying high-risk loans to mitigate financial risks, we recommend utilizing the logistic regression model trained on the resampled data for making loan status predictions. This model effectively balances the trade-off between precision and recall for both healthy and high-risk loans, making it well-suited for practical deployment in real-world scenarios.
