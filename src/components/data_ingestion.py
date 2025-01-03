import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.logging.logger import logging
from src.exception.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str =os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    row_data_path: str = os.path.join("artifacts",'row.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Enter the Data Ingestion method or component")
        try:
            df = pd.read_csv("notebook\Datasets\cleandata.csv")

            logging.info("Read data as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.row_data_path,index=False,header=True)

            logging.info("train test split initiate")
            train_set,test_set = train_test_split(df,test_size=0.3,random_state = 0)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
    

if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()