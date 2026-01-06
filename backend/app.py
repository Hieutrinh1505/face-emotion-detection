from typing import Literal
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn

try:
    # When running as a package (e.g., from Docker or project root)
    from backend.utils import load_model, transform_image, face_model_inference, emotion_model_inference
    from backend.models import EmotionPredictionResponse, FaceDetectionResponse
except ImportError:
    # When running directly from backend directory
    from utils import load_model, transform_image, face_model_inference, emotion_model_inference
    from models import EmotionPredictionResponse, FaceDetectionResponse

app = FastAPI(title="Face Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]



@app.post("/infer", response_model=EmotionPredictionResponse | FaceDetectionResponse)
async def infer(
    file: UploadFile = File(...),
    model_type: Literal["face", "emotion"] = "emotion",
):
    """
    Perform emotion detection on an uploaded face image.

    - **file**: Image file (JPEG, PNG)
    - **model_type**: Type of model to use ("face" or "emotion")
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        model = load_model(model_type)
        if model_type == "face":
            boxes = face_model_inference(Image.open(file.file), model)
            return FaceDetectionResponse(boxes=boxes)
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_tensor = transform_image(image)
        
        raw_probs = emotion_model_inference(input_tensor, model)
        return EmotionPredictionResponse(
            probabilities={
                emotion: float(prob)
                for emotion, prob in zip(EMOTIONS, raw_probs)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
