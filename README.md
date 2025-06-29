# 🔍 Model Evaluation and Hyperparameter Tuning
# OUTPUT:
Initial Model Evaluation:

--- Logistic Regression ---
              precision    recall  f1-score   support

           0       0.97      0.91      0.94        43
           1       0.95      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

--- Decision Tree ---
              precision    recall  f1-score   support

           0       0.93      0.93      0.93        43
           1       0.96      0.96      0.96        71

    accuracy                           0.95       114
   macro avg       0.94      0.94      0.94       114
weighted avg       0.95      0.95      0.95       114

--- Random Forest ---
              precision    recall  f1-score   support

           0       0.98      0.93      0.95        43
           1       0.96      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114

--- SVM ---
              precision    recall  f1-score   support

           0       1.00      0.86      0.93        43
           1       0.92      1.00      0.96        71

    accuracy                           0.95       114
   macro avg       0.96      0.93      0.94       114
weighted avg       0.95      0.95      0.95       114


Best Parameters using GridSearchCV (Decision Tree):
{'max_depth': 5, 'min_samples_split': 10}

Best Parameters using RandomizedSearchCV(Random Forest):
{'max_depth': 9, 'min_samples_split': 5, 'n_estimators': 24}

Final Evaluation of Best Models:

--- Tuned Decision Tree ---
Accuracy:  0.9298
Precision: 0.9437
Recall:    0.9437
F1-Score:  0.9437

--- Tuned Random Forest ---
Accuracy:  0.9561
Precision: 0.9583
Recall:    0.9718
F1-Score:  0.9650
