import pandas as pd

import os
import sys
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import *
from exceptions import CustomException
from logger import logging
from data_ingestion_transformation import *
from build_model import *

# pd.set_option('display.max_columns',100)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib


class predict_model():
    
    def __init__(self):
        self.folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def train_models(self):
        # get data to train models
        obj = build_model()
        X_train, y_train, X_test, y_test = obj.prepare_for_model()
        
        # model parameters

        model_dict = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=10000),
                'params': {
                    'C': uniform(0.01, 10),
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                },
            },

            

            'DecisionTree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
            },

            'RandomForest': {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                },
            },

            'KNN': {
                'model':  KNeighborsClassifier(),
                'params': {
                    'n_neighbors': randint(3, 20),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
            },

            'AdaBoost': {
                'model': AdaBoostClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                },
            },

            'GradientBoosting': {
                'model': GradientBoostingClassifier(),
                'params': {
                    'n_estimators': [100, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 3],
                    'subsample': [0.8, 1.0],
                    'max_features': ['sqrt', 'log2']
                },
            },

            'XGBoost': {
                'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': randint(100, 500),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 10),
                    'subsample': uniform(0.5, 0.5),
                    'colsample_bytree': uniform(0.5, 0.5),
                    'reg_alpha': uniform(0, 1),
                    'reg_lambda': uniform(0, 1)
                },
            },
        }

        

        best_models = {}

        for name, config in model_dict.items():
            print(f"Training: {name}")
            search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['params'],
                n_iter=20,
                cv=5,
                scoring='accuracy',
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
            search.fit(X_train, y_train)
            best_models[name] = search.best_estimator_
            y_pred = search.predict(X_test)
            print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            # print(classification_report(y_test, y_pred))
            joblib.dump(best_models[name], rf'{self.folder_path}\models\{name}_cricket_model.pkl')



if __name__=='__main__':
    obj = predict_model()
    obj.train_models()
