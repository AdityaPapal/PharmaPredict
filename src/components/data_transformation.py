import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,OneHotEncoder

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.utils import save_object
from src.components.data_ingestion import DataIngestion
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # This function is responsible for data transformation based on differnt data and its types 
        try:
            numerical_col = ['Age', 'Na_to_K']
            categorical_col = ['Sex', 'BP', 'Cholesterol']

            sex_cate = ["F","M"]
            BP_cate = ['LOW','NORMAL','HIGH']
            cholestrol_cate = ['NORMAL','HIGH']
            
            
            logging.info("Initiate Pipelines")

            logging.info("Start numerical pipeline")
            num_pipline = Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numberical pipeline completed")

            logging.info("Start categorical piplines")
            categorical_pipeline = Pipeline(
                steps=[
                    ('ordinalencoder',OrdinalEncoder(categories=[sex_cate,BP_cate,cholestrol_cate])),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info("Complete categorical pipeline")
            
            logging.info(f"categorical columns: {categorical_col}")
            logging.info(f"Numberical columns: {numerical_col}")
            
            logging.info("Initiate preprocessor")
            preprocessor = ColumnTransformer([
                ("num_pipline",num_pipline,numerical_col),
                ('categorical_pipeline',categorical_pipeline,categorical_col)
               
            ])
            logging.info("complete the preprocessor")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(train_df.columns)
            print(test_df.columns)
            logging.info("Read train and test data successfully")


            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()

            target_col_name ="Drug"
            drop_col = [target_col_name]

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            input_feature_train_df = train_df.drop(columns=drop_col,axis=1)
            target_feature_train_df=train_df[target_col_name]

            input_feature_test_df=test_df.drop(columns=drop_col,axis=1)
            target_feature_test_df=test_df[target_col_name]
            
            ## Trnasformating using preprocessor obj
            print(input_feature_train_df)
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Save preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
     obj=DataIngestion()
     train_data_path,test_data_path=obj.initiate_data_ingestion()
     data_transformation = DataTransformation()
     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)        