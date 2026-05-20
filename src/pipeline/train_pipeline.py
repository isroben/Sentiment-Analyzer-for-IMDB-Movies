import sys
import os

sys.path.append('/home/roben/Codes/Sentiment Analysis')

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

logger = get_logger(__name__)


def training_pipeline():
    try:
        logger.info("Training Pipeline started.")

        # ── Step 1: Data Ingestion ─────────────────────────────────────────────
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiateDataIngestion()   # returns two CSV paths
        logger.info("Data ingestion completed.")

        # ── Step 2: Data Transformation ───────────────────────────────────────
        transform = DataTransformation()
        X_train, X_test, y_train, y_test = transform.initiateDataTransformation(
            train_data_path, test_data_path
        )
        logger.info(f"Data transformation completed. X_train: {X_train.shape} | X_test: {X_test.shape}")

        # ── Step 3: Model Training ─────────────────────────────────────────────
        trainer = ModelTrainer()
        final_model, best_model_name, model_report = trainer.initiate_model_trainer(
            X_train, y_train, X_test, y_test
        )
        logger.info("Model training completed.")

        # ── Summary ───────────────────────────────────────────────────────────
        best = model_report[best_model_name]
        logger.info(f"Best Model   : {best_model_name}")
        logger.info(f"Test F1      : {best['test_f1']:.4f}")
        logger.info(f"Test Accuracy: {best['test_accuracy']:.4f}")
        logger.info(f"Best Params  : {best['best_params']}")

        print("\n================ TRAINING COMPLETED =================")
        print(f"  Best Model     : {best_model_name}")
        print(f"  Test Accuracy  : {best['test_accuracy']:.4f}")
        print(f"  Test F1        : {best['test_f1']:.4f}")
        print(f"  Test Precision : {best['test_precision']:.4f}")
        print(f"  Test Recall    : {best['test_recall']:.4f}")
        print(f"  Overfit Gap    : {best['overfit_gap']:+.4f}")
        print(f"  Best Params    : {best['best_params']}")
        print(f"  Model saved at : artifacts/model.pkl")
        print("=====================================================\n")

        return final_model, best_model_name, model_report

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    training_pipeline()
