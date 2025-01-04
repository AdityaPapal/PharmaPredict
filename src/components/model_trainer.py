import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.utils import save_object
from src.utils.utils import evaluate_model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              BaggingClassifier,
                              ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              StackingClassifier,
                              VotingClassifier)

from sklearn.naive_bayes import CategoricalNB,GaussianNB, MultinomialNB

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


@dataclass
class ModelTraininerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTraininerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            support_vector = SVC(kernel='linear', max_iter=251)
            Kneighbars = KNeighborsClassifier(n_neighbors=20)
            NBclassifier1 = CategoricalNB()
            NBclassifier2 = GaussianNB()
            decision_tree = DecisionTreeClassifier(max_leaf_nodes=20)
            logistic = LogisticRegression(solver = 'liblinear', penalty = 'l1',max_iter=5000)
            random_forest = RandomForestClassifier(max_leaf_nodes=30) 
            ada_boost = AdaBoostClassifier(n_estimators = 20, )
            bagging = BaggingClassifier(estimator = DecisionTreeClassifier(),n_estimators = 20,max_samples = 0.8,oob_score = True)
            Gradient_boost = GradientBoostingClassifier(n_estimators = 20)
            
            base_learner = [
                ('support_vector', SVC(kernel='linear', max_iter=251,probability=True)),  # Enable probability for SVC
                ('logistic', LogisticRegression(solver = 'liblinear', penalty = 'l1',max_iter=5000)),
                ('random_forest', RandomForestClassifier(max_leaf_nodes=30)),
                ('Kneighbars',KNeighborsClassifier(n_neighbors=20))   
            ]
            stacking = StackingClassifier(estimators=base_learner, final_estimator=DecisionTreeClassifier(max_leaf_nodes=20))
            
            model = {
                #'Support Vector Classifier' : support_vector,
                'K-Neighbors Classifier'  : Kneighbars,
                'Decision Tree Classifier'  : decision_tree,
                'Logistic Regression'  : logistic,
                'Random Forest Classifier'  : random_forest,
                'AdaBoost Classifier': ada_boost,
                'Bagging Classifier' : bagging,
                'Stacking':stacking,
                'Gradient Boosting Classifier' : Gradient_boost,
            }

            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,
                                               x_test=x_test,y_test=y_test,models=model)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = model[best_model_name]


            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            # bagging.fit(x_train,y_train)
            # predicted = bagging.predict(x_test)
            # accuracy = accuracy_score(predicted,y_test)
            # print(accuracy)
            
            # print(input_data_reshaped)
            # prediction = stacking.predict(input_data_reshaped)
            prediction = best_model.predict(x_test)
            accuracy = accuracy_score(prediction,y_test)
            
            return accuracy
            # if prediction == 'DrugX':
            #     # print("DrugX")
            #     return 1
            # elif prediction == 'DrugY':
            #     # print("DrugY")
            #     return 2
            # elif prediction == 'DrugA':
            #     # print("DrugA")
            #     return 3
            # elif prediction == 'DrugB':
            #     # print("DrugB")
            #     return 4
            # else:
            #     # print("DrugC")
            #     return 'DrugC'


            # bagging.fit(x_train, y_train) 
            # y_pred = bagging.predict(x_test)
            # accuracy = accuracy_score(y_test, y_pred)
            # precision = precision_score(y_test, y_pred,average="macro")
            # recall = recall_score(y_test, y_pred,average="macro")
            # F1_score = f1_score(y_test, y_pred,average="macro")
            # return accuracy
            # print("Accuracy   :", accuracy)
            # print("Precision  :", precision)
            # print("Recall     :", recall)
            # print("F1-score   :", F1_score)
            # logging.info(f"Accuracy:{accuracy} precision:{precision},recall:{recall}")

            
        except Exception as e:
            raise CustomException(e,sys)