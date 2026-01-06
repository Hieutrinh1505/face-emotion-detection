"""
Real-time Emotion Detection Web Application using Streamlit.

Usage:
    streamlit run app.py
"""

import numpy as np
import requests
import io

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import threading
import streamlit as st
import pandas as pd
import time
from PIL import Image

# Backend API URL
API_URL = "http://localhost:8000"

# Emotion icons mapping
EMOTION_ICONS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}


class MyProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.emotion_probs = {e: 0.0 for e in EMOTION_ICONS.keys()}
        self.lock = threading.Lock()
        self.prob_buffer = []
        self.buffer_size = 5
        self.frame_count = 0
        self.inference_interval = 5
        self.cached_boxes = None
        self.had_face = False

    def _call_infer_api(self, image: Image.Image, model_type: str) -> dict | None:
        try:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            response = requests.post(
                f"{API_URL}/infer",
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
                params={"model_type": model_type},
                timeout=5,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None

    def _detect_faces(self, img_rgb: np.ndarray) -> list | None:
        result = self._call_infer_api(Image.fromarray(img_rgb), "face")
        return result.get("boxes") if result else None

    def _get_emotion_probs(self, face_img: np.ndarray) -> dict | None:
        result = self._call_infer_api(Image.fromarray(face_img), "emotion")
        return result.get("probabilities") if result else None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.frame_count += 1
        run_inference = self.frame_count % self.inference_interval == 0

        if run_inference:
            boxes = self._detect_faces(img_rgb)
            if boxes and len(boxes) > 0:
                self.cached_boxes = boxes
                self.had_face = True
            else:
                if self.had_face:
                    self.prob_buffer = []
                    with self.lock:
                        self.emotion_probs = {e: 0.0 for e in EMOTION_ICONS.keys()}
                self.cached_boxes = None
                self.had_face = False

        boxes = self.cached_boxes

        if boxes and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)

                face = img_rgb[y1:y2, x1:x2]

                if face.size > 0 and run_inference:
                    probs = self._get_emotion_probs(face)
                    if probs:
                        self.prob_buffer.append(list(probs.values()))
                        if len(self.prob_buffer) > self.buffer_size:
                            self.prob_buffer.pop(0)

                        avg_probs = np.mean(self.prob_buffer, axis=0)
                        with self.lock:
                            for i, emotion in enumerate(probs.keys()):
                                self.emotion_probs[emotion] = float(avg_probs[i])

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display highest emotion
                with self.lock:
                    probs_copy = self.emotion_probs.copy()

                if self.prob_buffer:
                    max_emotion = max(probs_copy, key=probs_copy.get)
                    confidence = probs_copy[max_emotion]

                    if confidence > 0.1:
                        label = f"{max_emotion}: {confidence:.0%}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                        cv2.rectangle(img, (x1, y1 - 35), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                        cv2.putText(img, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Page config
st.set_page_config(page_title="Emotion Detection", layout="wide")

st.title("Real-time Emotion Detection")

# Side by side layout - video left, chart right
col1, col2 = st.columns([1, 1])

with col1:
    ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=MyProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}},
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

with col2:
    st.subheader("Emotion Probabilities")
    chart_placeholder = st.empty()

    if ctx.video_processor:
        while ctx.state.playing:
            with ctx.video_processor.lock:
                current_probs = ctx.video_processor.emotion_probs.copy()

            # Create DataFrame with icons
            df = pd.DataFrame({
                "Emotion": [f"{EMOTION_ICONS[e]} {e.capitalize()}" for e in current_probs.keys()],
                "Probability": list(current_probs.values()),
            })
            df = df.sort_values("Probability", ascending=True)

            chart_placeholder.bar_chart(df.set_index("Emotion"), horizontal=True, height=400)
            time.sleep(0.3)

# Sidebar
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Click **START** to begin
    2. Position your face in frame
    3. See real-time predictions!

    **Tips:**
    - Good lighting helps
    - Face the camera directly
    """)
