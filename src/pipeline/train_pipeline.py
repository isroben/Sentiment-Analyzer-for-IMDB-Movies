import sys
import os
sys.path.append('/home/roben/Codes/Sentiment Analysis')
# print(sys.path)


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

def training_pipeline():
    try:
        logger.info("Training Pipeline started.")

        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiateDataIngestion()

        logger.info("Data ingestion completed.")

        transfomer = DataTransformation()
        train_array, test_array = transfomer.initiateDataTransformation(train_data_path, test_data_path)

        logger.info("Data transformation completed.")

        print(type(train_array))

    
    except Exception as e:
        raise CustomException(e, sys)
    

training_pipeline()
    
