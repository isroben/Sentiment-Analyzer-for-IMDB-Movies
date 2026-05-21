import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schema import PredictRequest, PredictResponse, HealthResponse
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load prediction pipeline once at startup — not on every request."""
    global pipeline
    try:
        pipeline = PredictPipeline()
        logger.info("Prediction pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load prediction pipeline: {e}")
        logger.warning("API starting without a loaded model. Train first.")
    yield
    # Cleanup on shutdown (nothing needed here)


app = FastAPI(
    title="Sentiment Classifier API",
    description="Classifies movie reviews as positive or negative.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    result = pipeline.predict(request.text)

    return PredictResponse(
        label=result['label'],
        confidence=result['confidence'] if result['confidence'] is not None else 0.0,
        label_id=result['label_id'],
    )


@app.get("/")
def root():
    return {"message": "Sentiment Classifier API. Visit /docs for usage."}
