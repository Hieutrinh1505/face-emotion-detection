from pydantic import BaseModel
from typing import Literal, Any


class InferenceRequest(BaseModel):
    """Request model specifying which detector to use."""
    model_type: Literal["face", "emotion"]


class FaceDetectionResponse(BaseModel):
    """Response for face detection - returns bounding boxes."""
    boxes: Any  # [[x1, y1, x2, y2], ...]


class EmotionPredictionResponse(BaseModel):
    """Response for emotion detection - returns probabilities."""
    probabilities: dict[str, float]
