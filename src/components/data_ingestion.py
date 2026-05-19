import os
import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    source_data_path: str = r"data/"
    train_data_path: str = os.path.join('data/processed', 'train.csv')
    test_data_path: str = os.path.join('data/processed', 'test.csv')
    raw_data_path: str = os.path.join('data/raw', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiateDataIngestion(self):
        logger.info("Starting data ingestion process.")

        try:
            if not os.path.exists(self.config.source_data_path):
                raise CustomException(f"Source data file does not exits: {self.config.source_data_path}", sys)
            
            dataset = pd.read_csv(self.config.source_data_path)
            logger.info("Dataset loaded successfully.")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

            logger.info("Train-Test split completed.")

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=False)

            logger.info("Data ingestion completed.")

            return self.config.train_data_path, self.config.test_data_path
        

        except Exception as e:
            raise CustomException(e, sys)