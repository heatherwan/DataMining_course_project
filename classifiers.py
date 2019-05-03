#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix



classifiers = {
#     "Naive Bayes" : GaussianNB(),
    "K Nerest Neighbors": KNeighborsClassifier(10),
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM": SVC(gamma=2, C=1),
    "Logistic Regression": LogisticRegression( solver='lbfgs', max_iter = 500 ), 
#     "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "AdaBoost" : AdaBoostClassifier(),
    "XGBoost": xgb.XGBClassifier()
}

def profit_scorer(y, y_pred):
#     print(confusion_matrix(y, y_pred))
    profit_matrix = {(0,0): 0, (0,1): -5, (1,0): -25, (1,1): 5}
    return sum(profit_matrix[(pred, actual)] for pred, actual in zip(y_pred, y))

def evaluate_classification(X, y):
    cv = StratifiedKFold(n_splits=10, random_state=42)
    profit_scoring = make_scorer(profit_scorer, greater_is_better=True)
    
    for name, clf in classifiers.items():
#         print(cross_validate(clf, X, y=y, cv=cv, scoring=profit_scoring)['test_score'])
        result = sum(cross_validate(clf, X, y=y, cv=cv, scoring=profit_scoring)['test_score'])
        print(f"{name}: test core = {result} ")
        
