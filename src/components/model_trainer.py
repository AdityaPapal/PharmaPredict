import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.utils import evaluate_model, save_object
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import SMOTE

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
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)

            logging.info(f"Class Distribution After Balancing: {Counter(y_train)}")

            support_vector = SVC(kernel='linear', max_iter=251,class_weight='balanced')
            Kneighbars = KNeighborsClassifier(n_neighbors=20)
            NBclassifier1 = CategoricalNB()
            NBclassifier2 = GaussianNB()
            decision_tree = DecisionTreeClassifier(max_leaf_nodes=20,class_weight='balanced')
            logistic = LogisticRegression(solver = 'liblinear', penalty = 'l1',max_iter=5000,class_weight='balanced')
            random_forest = RandomForestClassifier(max_leaf_nodes=30,class_weight='balanced') 
            ada_boost = AdaBoostClassifier(n_estimators = 20)
            Gradient_boost = GradientBoostingClassifier(n_estimators = 20)
            base_learner = [
                ('support_vector', SVC(kernel='linear', max_iter=251,probability=True,class_weight='balanced')),  # Enable probability for SVC
                ('logistic', LogisticRegression(solver = 'liblinear', penalty = 'l1',class_weight='balanced',max_iter=5000)),
                ('random_forest', RandomForestClassifier(max_leaf_nodes=30,class_weight='balanced')),
                ('Kneighbars',KNeighborsClassifier(n_neighbors=20))
            ]
            bagging = BaggingClassifier(estimator = base_learner,n_estimators = 20,max_samples = 0.8,oob_score = True)

            stacking = StackingClassifier(estimators=base_learner, final_estimator=DecisionTreeClassifier(max_leaf_nodes=20))
            

            voting_clf = VotingClassifier(estimators=base_learner, voting='soft')
            voting_clf.fit(x_train, y_train)
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
                'Voting Classifier':voting_clf
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