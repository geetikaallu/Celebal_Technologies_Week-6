#Import required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')

data=load_breast_cancer()
X=data.data
y=data.target

#Split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Define models
models={
    "Logistic Regression":LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "SVM":SVC()
}

#Evaluate each model
print("Initial Model Evaluation:\n")
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"--- {name} ---")
    print(classification_report(y_test,y_pred))

#Hyperparameter tuning: GridSearchCV on Decision Tree
param_grid={
    'max_depth':[3,5,10,None],
    'min_samples_split':[2,5,10]
}
grid_search=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5,scoring='f1')
grid_search.fit(X_train,y_train)

print("\nBest Parameters using GridSearchCV (Decision Tree):")
print(grid_search.best_params_)

#Hyperparameter tuning:RandomizedSearchCV on Random Forest
param_dist={
    'n_estimators':randint(10,100),
    'max_depth':randint(3,20),
    'min_samples_split':randint(2,11)
}
random_search=RandomizedSearchCV(RandomForestClassifier(),param_distributions=param_dist,n_iter=10,cv=5,scoring='f1',random_state=42)
random_search.fit(X_train,y_train)

print("\nBest Parameters using RandomizedSearchCV(Random Forest):")
print(random_search.best_params_)

#Evaluate best models after tuning
best_dt=grid_search.best_estimator_
best_rf=random_search.best_estimator_

print("\nFinal Evaluation of Best Models:")

for name,model in [("Tuned Decision Tree",best_dt), ("Tuned Random Forest",best_rf)]:
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    prec=precision_score(y_test,y_pred)
    rec=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
