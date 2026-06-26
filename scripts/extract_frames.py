import cv2, os, argparse, sys
from dataclasses import dataclass
import mediapipe as mp

@dataclass
class ExtractionConfig:
    face_size: int = 96
    padding_ratio: float = 0.20
    min_detection_confidence: float = 0.6
    model_selection: int = 0

class FaceExtractor:
    def __init__(self, config: ExtractionConfig = ExtractionConfig()):
        self.config = config
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=config.model_selection,
            min_detection_confidence=config.min_detection_confidence
        )
    
    def extract(self, video_path: str, label: str, fps_sample: int = 1):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        out_dir = os.path.join("data", "processed", label)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if not vid_fps or vid_fps <= 0:
            vid_fps = 20
            print(f"  Warning: Could not detect FPS, using default {vid_fps}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(vid_fps / fps_sample))
        saved, skipped, frame_idx = 0, 0, 0
        base = os.path.splitext(os.path.basename(video_path))[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % interval != 0:
                continue

            face = self._extract_face_from_frame(frame)
            if face is None:
                skipped += 1
                continue

            face = cv2.resize(face, (self.config.face_size, self.config.face_size))
            fname = os.path.join(out_dir, f"{label}_{base}_{frame_idx:05d}.jpg")
            cv2.imwrite(fname, face)
            saved += 1

        cap.release()
        print(f"  {os.path.basename(video_path)}: saved={saved}  skipped(no face)={skipped}")
        return saved, skipped
    
    def _extract_face_from_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_detector.process(rgb)

        if not result.detections:
            return None

        det = result.detections[0]
        bb = det.location_data.relative_bounding_box
        H, W = frame.shape[:2]

        pad = self.config.padding_ratio
        x1 = max(0, int((bb.xmin - pad * bb.width) * W))
        y1 = max(0, int((bb.ymin - pad * bb.height) * H))
        x2 = min(W, int((bb.xmin + (1 + pad) * bb.width) * W))
        y2 = min(H, int((bb.ymin + (1 + pad) * bb.height) * H))

        face = frame[y1:y2, x1:x2]
        return face if face.size > 0 else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract face crops from video for focus detection")
    ap.add_argument("--video", required=True, help="Path to input video file")
    ap.add_argument("--label", required=True, choices=["focused", "not_focused"], 
                    help="Classification label for the video")
    ap.add_argument("--fps", type=int, default=1, 
                    help="Target frames per second to extract (default: 1)")
    ap.add_argument("--face-size", type=int, default=96,
                    help="Output face crop size in pixels (default: 96)")
    ap.add_argument("--padding", type=float, default=0.20,
                    help="Padding ratio around detected face (default: 0.20)")
    args = ap.parse_args()
    
    try:
        config = ExtractionConfig(
            face_size=args.face_size,
            padding_ratio=args.padding
        )
        extractor = FaceExtractor(config)
        extractor.extract(args.video, args.label, args.fps)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)