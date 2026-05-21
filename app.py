"""
Entry point for the Sentiment Analysis application.

Usage:
    Flask web app:  python app.py
    FastAPI:        uvicorn src.api.main:app --reload
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)

# Load the prediction pipeline once at startup
pipeline = None

try:
    pipeline = PredictPipeline()
    logger.info("Prediction pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load prediction pipeline: {e}")
    logger.warning("Starting without a loaded model. Train first.")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded. Run training first.'}), 503

    data = request.get_json()
    review_text = data.get('review', '')

    if not review_text.strip():
        return jsonify({'error': 'Review text is empty.'}), 400

    try:
        result = pipeline.predict(review_text)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

    return jsonify({
        'prediction': result['label'],
        'label_id': result['label_id'],
        'confidence': round(result['confidence'] * 100, 2) if result['confidence'] else None,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
