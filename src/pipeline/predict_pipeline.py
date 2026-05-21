import os
import sys
import numpy as np

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import load_obj
from src.components.data_transformation import TextCleaner

logger = get_logger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts')


class PredictPipeline:
    def __init__(self):
        try:
            vectorizer_path = os.path.join(ARTIFACTS_DIR, 'vectorizer.pkl')
            model_path = os.path.join(ARTIFACTS_DIR, 'model.pkl')
            encoder_path = os.path.join(ARTIFACTS_DIR, 'encoder.pkl')

            self.vectorizer = load_obj(vectorizer_path)
            self.model = load_obj(model_path)
            self.encoder = load_obj(encoder_path)
            self.cleaner = TextCleaner()

            logger.info("Prediction pipeline initialized successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, review_text: str) -> dict:
        """
        End-to-end prediction for a single review string.

        Returns:
            dict with keys: label, label_id, confidence (if available)
        """
        try:
            # 1. Clean text (same preprocessing as training)
            cleaned = self.cleaner.clean(review_text)
            logger.info(f"Cleaned text: {cleaned[:100]}...")

            # 2. Vectorize (Word2Vec document vector)
            features = self.vectorizer.transform([cleaned])
            logger.info(f"Feature vector shape: {features.shape}")

            # 3. Model prediction
            label_id = int(self.model.predict(features)[0])

            # 4. Confidence (not all models support predict_proba)
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(features)[0]
                    confidence = float(proba[label_id])
                except Exception:
                    pass

            # 5. Decode label
            label = self.encoder.inverse_transform([label_id])[0]

            result = {
                'label': label,
                'label_id': label_id,
                'confidence': round(confidence, 4) if confidence else None,
            }
            logger.info(f"Prediction result: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Wraps a raw review string and converts it into the format
    expected by PredictPipeline."""

    def __init__(self, review: str):
        self.review = review

    def get_review_text(self) -> str:
        return self.review