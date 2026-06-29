"""
inference_engine.py -- Lightweight webcam inference.

Opens camera, grabs one frame every INFERENCE_INTERVAL seconds,
runs MediaPipe face crop + CNN model, yields (label, confidence, timestamp).
Camera is released the moment the context manager exits — zero resource
usage while the scheduler is sleeping between check windows.
"""

import cv2, time, os
import torch
from torchvision import transforms
import mediapipe as mp
import mlflow.pytorch
from dotenv import load_dotenv

load_dotenv(r"config\.env")

INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "2"))
MLFLOW_URI         = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME         = os.getenv("MODEL_NAME", "FocusClassifier")
CLASSES            = ["focused", "not_focused"]

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_mp_face       = mp.solutions.face_detection
_face_detector = _mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)


def load_model():
    """Load the current Production model from the MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    uri   = f"models:/{MODEL_NAME}/Production"
    print(f"Loading model from registry: {uri}")
    model = mlflow.pytorch.load_model(uri)
    model.eval()
    print("Model ready.")
    return model


def _crop_face(frame):
    """Return a 96x96 BGR face crop, or None if no face detected."""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _face_detector.process(rgb)
    if not result.detections:
        return None

    det  = result.detections[0]
    bb   = det.location_data.relative_bounding_box
    H, W = frame.shape[:2]
    pad  = 0.20

    x1 = max(0, int((bb.xmin - pad * bb.width) * W))
    y1 = max(0, int((bb.ymin - pad * bb.height) * H))
    x2 = min(W, int((bb.xmin + (1 + pad) * bb.width) * W))
    y2 = min(H, int((bb.ymin + (1 + pad) * bb.height) * H))

    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (96, 96)) if face.size > 0 else None


def predict_frame(model, face_bgr) -> tuple:
    """Returns (class_label: str, confidence: float)."""
    rgb    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return CLASSES[idx], float(probs[idx])


class InferenceSession:
    """
    Context manager — opens camera on enter, releases on exit.

    Usage:
        with InferenceSession(model, duration=60) as session:
            for label, confidence, timestamp in session:
                ...
    """

    def __init__(self, model, duration: float = None):
        self.model    = model
        self.duration = duration
        self._cap     = None
        self._stop    = False

    def stop(self):
        self._stop = True

    def __enter__(self):
        self._cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check index in cv2.VideoCapture().")
        return self

    def __exit__(self, *_):
        if self._cap:
            self._cap.release()
        self._cap = None

    def __iter__(self):
        start = time.time()
        while not self._stop:
            if self.duration and (time.time() - start) > self.duration:
                break

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            face = _crop_face(frame)
            if face is not None:
                label, conf = predict_frame(self.model, face)
            else:
                # No face = treat as not_focused (walked away / turned away fully)
                label, conf = "not_focused", 0.0

            yield label, conf, time.time()
            time.sleep(INFERENCE_INTERVAL)
            
            
            
#Todo : Check leaks using my own https://github.com/shah-shuja-dev/process-leak-detector