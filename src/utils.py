#will have common functions which we will use in the projects

import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            
            # Retrieve the hyperparameters for the current model
            model_params = param.get(model_name, {})
            
            # Perform Grid Search CV for hyperparameter tuning
            gs = GridSearchCV(estimator=model, param_grid=model_params, cv=3, scoring='r2', n_jobs=-1, verbose=2)
            gs.fit(X_train, y_train)
            
            # Get the best estimator and evaluate on the test set
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
            
            # Calculate R^2 score
            model_score = r2_score(y_test, y_pred)
            report[model_name] = model_score

            logging.info(f"{model_name} achieved R^2 score: {model_score}")
        
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
