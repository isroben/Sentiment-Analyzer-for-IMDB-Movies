import os
import sys
from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import save_object
from src.components.model_evaluation import evaluate_model

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


logger = get_logger(__name__)

@dataclass
class modelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()

    def initiateModelTrainer(self, train_arr, test_arr):
        try:
            models = {
                "Random Forest" : RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-NN": KNeighborsClassifier(),
                "XGB": XGBClassifier(),
                "Adaboost": AdaBoostClassifier()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },

                'Random Forest': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2',1],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                'Gradient Boosting': {
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['auto', 'squrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                'Linear Regression':{},
                'XGBRegressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },

                'Support Vector machine': {
                    'C':[0.1,0.3,0.5,1,10,20],
                    'kernel':['rbf', 'linear']
                },

                'AdaBoost Regressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            report, trained_models, best_model_name = evaluate_model()
        except Exception as e:
            raise CustomException(e, sys)