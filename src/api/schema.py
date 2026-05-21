"""
Pydantic schemas for request and response validation.
FastAPI uses these to auto-generate docs and validate inputs.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The review text to classify.",
        examples=["This movie was absolutely brilliant!"],
    )


class PredictResponse(BaseModel):
    label: str        = Field(..., description="'positive' or 'negative'")
    confidence: float = Field(..., description="Model confidence (0.0 – 1.0)")
    label_id: int     = Field(..., description="0 = negative, 1 = positive")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
