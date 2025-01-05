import pandas as pd
import sys
import os
from src.exception.exception import CustomException
from src.utils.utils import load_model
from src.logging.logger import logging

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_model(preprocessor_path)
            model=load_model(model_path)

            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                Age:int,
                Sex:str,
                BP:str,
                Cholesterol:str,
                Na_to_K:float):
        self.Age = Age
        self.Sex = Sex
        self.BP = BP
        self.Cholesterol = Cholesterol
        self.Na_to_K = Na_to_K


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'Sex':[self.Sex],
                'BP':[self.BP],
                'Cholesterol':[self.Cholesterol],
                'Na_to_K':[self.Na_to_K]
            }
            data = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return data
        
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)