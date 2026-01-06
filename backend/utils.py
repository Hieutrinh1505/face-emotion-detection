import os
from facenet_pytorch import MTCNN
import torch
from torchvision.transforms import v2

# GLOBAL VARIABLES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_DETECTOR = MTCNN(
    keep_all=True,
    device=DEVICE,
    min_face_size=80,
    thresholds=[0.7, 0.8, 0.8],
    post_process=True,
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "efficientnet_b0_adamw_256x256_2.pt")
EMOTION_DETECTOR = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
).eval()

IMAGE_TRANSFORM = v2.Compose(
    [
        v2.Resize((256, 256)),
        v2.ToTensor(),
    ]
)


def load_model(detector_type):
    if detector_type == "face":
        return FACE_DETECTOR

    return EMOTION_DETECTOR


def transform_image(image) -> torch.Tensor:
    """Transform input image for emotion detection model."""
    # Ensure image is RGB (3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)


def emotion_model_inference(input_tensor, model):
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        raw_probs = probabilities.squeeze().cpu().numpy()
        return raw_probs


def face_model_inference(image, model):
    """Perform face detection and return bounding boxes as list."""
    import numpy as np
    from PIL import Image as PILImage

    # Convert PIL Image to RGB numpy array
    if isinstance(image, PILImage.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = image

    boxes, _ = model.detect(img_rgb)

    # Convert numpy array to list for JSON serialization
    if boxes is not None:
        return boxes.tolist()
    return []
