import os
import sys
import dill
from typing import Dict, Any
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger  = get_logger(__name__)

def save_object(file_path: str, obj: Any):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj, protocol=dill.HIGHEST_PROTOCOL)
        logger.info(f"Object saved successfully at: {file_path}")


    except Exception as e:
        raise CustomException(e, sys)
    
def load_obj(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        logger.info(f"Object loaded successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)