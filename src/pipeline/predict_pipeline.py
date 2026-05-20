import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.utils import load_obj


class PredictPipeline:
    def __init__(self):
        try:
            self.preprocessor = load_obj(r'artifacts/')
            self.model = load_obj(r'')

        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, features):
        try:
            data_cleaned = self.preprocessor.transform(features)
            print(f"Data scaled after transformation is: \n {data_cleaned}")
            preds = self.model.predict(data_cleaned)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                'age': [self.data]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)