import os
import sys
import dill
import pandas as pd
import numpy as np
from src.exception.exception import CustomException
from src.logging.logger import logging

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Train Model
            model.fit(x_train,y_train)

            # Predict Testing Data
            y_test_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test_pred,y_test)
            precision = precision_score(y_test_pred,y_test,average="macro")
            matrix = confusion_matrix(y_test_pred,y_test)
            report[list(models.keys())[i]] = accuracy

            return report

    except Exception as e:
        logging.info("Exception occurs in evaluate model function in utils")
        raise CustomException(e,sys)