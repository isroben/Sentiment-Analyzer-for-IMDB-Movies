import os
import sys
from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import save_object
from src.components.model_evaluation import evaluate_models

# Classification models only — no Regressors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = get_logger(__name__)

SUBSET_SIZE  = 10000   # rows used for model selection + hyperparam search
FULL_SIZE    = 50000   # rows used for final retrain of the winning model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        """
        Strategy:
          1. Evaluate all models on a 10k subset  → find best model + best params
          2. Retrain that one model on full 50k data with best params
          3. Save the final model
        """
        try:
            # ── Step 1: Subset for evaluation ─────────────────────────────────
            logger.info(f"Sampling {SUBSET_SIZE} rows for model selection phase.")

            # Stratified sample to keep pos/neg balance in the subset
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=SUBSET_SIZE, random_state=42)
            subset_idx, _ = next(sss.split(X_train, y_train))

            X_sub = X_train[subset_idx]
            y_sub = y_train[subset_idx]

            # Subset test set too (2k) so GridSearch feedback loop is fast
            sss_test = StratifiedShuffleSplit(n_splits=1, train_size=2_000, random_state=42)
            test_idx, _ = next(sss_test.split(X_test, y_test))
            X_sub_test = X_test[test_idx]
            y_sub_test  = y_test[test_idx]

            logger.info(f"Subset — Train: {X_sub.shape} | Test: {X_sub_test.shape}")

            # ── Models (classifiers) ───────────────────────────────────────────
            models = {
                "Logistic Regression":    LogisticRegression(max_iter=1000),
                "Random Forest":          RandomForestClassifier(),
                "Decision Tree":          DecisionTreeClassifier(),
                "Gradient Boosting":      GradientBoostingClassifier(),
                "K-Neighbors":            KNeighborsClassifier(),
                "Support Vector Machine": SVC(),
                "XGBoost":                XGBClassifier(eval_metric='logloss', verbosity=0),
                "AdaBoost":               AdaBoostClassifier(),
            }

            # ── Param grids (classification-appropriate) ───────────────────────
            params = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                },
                "Random Forest": {
                    'n_estimators': [64, 128, 256],
                    'max_depth': [None, 10, 20],
                    'max_features': ['sqrt', 'log2']
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'max_features': ['sqrt', 'log2']
                },
                "Gradient Boosting": {
                    'n_estimators': [64, 128],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5]
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7, 11],
                    'weights': ['uniform', 'distance']
                },
                "Support Vector Machine": {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                },
                "XGBoost": {
                    'n_estimators': [64, 128],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            }

            # ── Run evaluation on subset ───────────────────────────────────────
            logger.info("Starting model evaluation on subset...")
            report, _, best_model_name = evaluate_models(
                X_sub, y_sub, X_sub_test, y_sub_test, models, params
            )

            best_params = report[best_model_name]['best_params']
            best_f1     = report[best_model_name]['test_f1']
            logger.info(f"Winner: {best_model_name} | Subset F1: {best_f1:.4f} | Params: {best_params}")

            # ── Step 2: Retrain winner on FULL data ────────────────────────────
            logger.info(f"Retraining '{best_model_name}' on full {X_train.shape[0]} training samples...")

            final_model = models[best_model_name]           # already has best_params set
            final_model.set_params(**best_params)           # ensure params are applied
            final_model.fit(X_train, y_train)               # full 40k train split

            # ── Evaluate final model on full test set ──────────────────────────
            from sklearn.metrics import accuracy_score, f1_score, classification_report
            y_final_pred = final_model.predict(X_test)

            final_accuracy = accuracy_score(y_test, y_final_pred)
            final_f1       = f1_score(y_test, y_final_pred, zero_division=0)

            print(f"\n{'='*55}")
            print(f"  FINAL MODEL: {best_model_name}")
            print(f"  Trained on : {X_train.shape[0]:,} samples")
            print(f"  Test Accuracy : {final_accuracy:.4f}")
            print(f"  Test F1       : {final_f1:.4f}")
            print(f"{'='*55}")
            print(classification_report(y_test, y_final_pred, target_names=['negative', 'positive']))

            logger.info(f"Final model — Accuracy: {final_accuracy:.4f} | F1: {final_f1:.4f}")

            # ── Step 3: Save ───────────────────────────────────────────────────
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            save_object(file_path=self.config.trained_model_file_path, obj=final_model)
            logger.info(f"Final model saved to: {self.config.trained_model_file_path}")

            return final_model, best_model_name, report

        except Exception as e:
            raise CustomException(e, sys)
