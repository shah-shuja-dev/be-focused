import cv2, os, argparse, sys, logging
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing MediaPipe with fallback for different versions
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    
    # Check if we're using the new API (0.10.x+) or old API (< 0.10)
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        MEDIAPIPE_NEW_API = True
        logger.info("Using MediaPipe new API (0.10.x+)")
    except (ImportError, AttributeError):
        try:
            # Old API fallback
            mp_face = mp.solutions.face_detection
            MEDIAPIPE_NEW_API = False
            logger.info("Using MediaPipe legacy API (< 0.10)")
        except AttributeError:
            MEDIAPIPE_AVAILABLE = False
            logger.warning("MediaPipe found but no compatible face detection API")
            
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_NEW_API = False
    logger.warning("MediaPipe not installed. Using OpenCV fallback detector.")

class FaceDetector:
    """Unified face detector supporting multiple backends"""
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.backend = None
        self.detector = None
        
        if MEDIAPIPE_AVAILABLE:
            if MEDIAPIPE_NEW_API:
                self._init_mediapipe_new(min_confidence)
            else:
                self._init_mediapipe_legacy(min_confidence)
        else:
            self._init_opencv_fallback()
    
    def _init_mediapipe_new(self, min_confidence: float):
        """Initialize MediaPipe 0.10.x+ detector"""
        try:
            base_options = mp_python.BaseOptions(model_asset_path=None)
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_confidence
            )
            self.detector = vision.FaceDetector.create_from_options(options)
            self.backend = "mediapipe_new"
            logger.info("Initialized MediaPipe face detector (new API)")
        except Exception as e:
            logger.warning(f"Failed to init MediaPipe new API: {e}. Falling back to OpenCV.")
            self._init_opencv_fallback()
    
    def _init_mediapipe_legacy(self, min_confidence: float):
        """Initialize MediaPipe < 0.10 detector"""
        try:
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_confidence
            )
            self.backend = "mediapipe_legacy"
            logger.info("Initialized MediaPipe face detector (legacy API)")
        except Exception as e:
            logger.warning(f"Failed to init MediaPipe legacy API: {e}. Falling back to OpenCV.")
            self._init_opencv_fallback()
    
    def _init_opencv_fallback(self):
        """Initialize OpenCV Haar Cascade as fallback"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                # Try alternative path
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise ValueError("Failed to load cascade classifier")
            
            self.backend = "opencv"
            logger.info("Initialized OpenCV Haar Cascade face detector (fallback)")
        except Exception as e:
            logger.error(f"Failed to initialize any face detector: {e}")
            raise RuntimeError("No face detection backend available. Install MediaPipe or ensure OpenCV is properly installed.")
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame and return bounding box (x, y, width, height)
        Returns None if no face detected
        """
        if self.backend == "mediapipe_new":
            return self._detect_mediapipe_new(frame)
        elif self.backend == "mediapipe_legacy":
            return self._detect_mediapipe_legacy(frame)
        elif self.backend == "opencv":
            return self._detect_opencv(frame)
        else:
            return None
    
    def _detect_mediapipe_new(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect using MediaPipe new API"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            
            if result.detections:
                # Get highest confidence detection
                best_det = max(result.detections, key=lambda d: d.score[0] if hasattr(d, 'score') else 1.0)
                bbox = best_det.bounding_box
                return (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        except Exception as e:
            logger.debug(f"MediaPipe new API detection failed: {e}")
        return None
    
    def _detect_mediapipe_legacy(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect using MediaPipe legacy API"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.detector.process(rgb)
            
            if result.detections:
                # Get highest confidence detection
                best_det = max(result.detections, key=lambda d: d.score[0])
                bb = best_det.location_data.relative_bounding_box
                H, W = frame.shape[:2]
                
                x = int(bb.xmin * W)
                y = int(bb.ymin * H)
                w = int(bb.width * W)
                h = int(bb.height * H)
                return (x, y, w, h)
        except Exception as e:
            logger.debug(f"MediaPipe legacy API detection failed: {e}")
        return None
    
    def _detect_opencv(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect using OpenCV Haar Cascade"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Return the largest face (closest to camera)
                largest = max(faces, key=lambda f: f[2] * f[3])
                return tuple(largest)
        except Exception as e:
            logger.debug(f"OpenCV detection failed: {e}")
        return None

class FaceExtractor:
    """Extract face crops from videos with configurable parameters"""
    
    def __init__(self, face_size: int = 96, padding_ratio: float = 0.20, 
                 min_confidence: float = 0.6):
        self.face_size = face_size
        self.padding_ratio = padding_ratio
        self.detector = FaceDetector(min_confidence=min_confidence)
    
    def extract(self, video_path: str, label: str, fps_sample: int = 1) -> Tuple[int, int]:
        """
        Extract face crops from video
        
        Args:
            video_path: Path to input video
            label: Classification label ('focused' or 'not_focused')
            fps_sample: Target frames per second to extract
        
        Returns:
            Tuple of (saved_count, skipped_count)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        out_dir = os.path.join("data", "processed", label)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if not vid_fps or vid_fps <= 0:
            vid_fps = 20
            logger.warning(f"Could not detect FPS, using default {vid_fps}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(vid_fps / fps_sample))
        
        video_name = Path(video_path).stem
        logger.info(f"Processing '{video_name}': {total_frames} frames @ {vid_fps:.1f}fps")
        
        saved, skipped, frame_idx = 0, 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Skip frames based on sampling interval
            if frame_idx % interval != 0:
                continue
            
            # Extract face
            face_crop = self._extract_face(frame)
            if face_crop is None:
                skipped += 1
                continue
            
            # Resize and save
            face_crop = cv2.resize(face_crop, (self.face_size, self.face_size))
            filename = f"{label}_{video_name}_{frame_idx:05d}.jpg"
            filepath = os.path.join(out_dir, filename)
            cv2.imwrite(filepath, face_crop)
            saved += 1
            
            # Progress update
            if saved % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                logger.debug(f"Progress: {progress:.1f}%")

        cap.release()
        logger.info(f"Finished '{video_name}': {saved} saved, {skipped} skipped")
        return saved, skipped
    
    def _extract_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract face region from frame with padding"""
        bbox = self.detector.detect(frame)
        if bbox is None:
            return None
        
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        pad = self.padding_ratio
        
        # Calculate padded coordinates
        x1 = max(0, int(x - pad * w))
        y1 = max(0, int(y - pad * h))
        x2 = min(W, int(x + (1 + pad) * w))
        y2 = min(H, int(y + (1 + pad) * h))
        
        face = frame[y1:y2, x1:x2]
        return face if face.size > 0 else None

def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops from videos for focus detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python extract_faces.py --video focused_clip.avi --label focused
  
  # Process with custom settings
  python extract_faces.py --video clip.avi --label focused --fps 2 --face-size 128
  
  # Batch processing (PowerShell)
  Get-ChildItem data\\raw\\focused_*.avi | ForEach-Object {
      python scripts\\extract_faces.py --video $_.FullName --label focused
  }
        """
    )
    
    parser.add_argument("--video", required=True, 
                       help="Path to input video file")
    parser.add_argument("--label", required=True, 
                       choices=["focused", "not_focused"],
                       help="Classification label")
    parser.add_argument("--fps", type=int, default=1,
                       help="Target FPS for extraction (default: 1)")
    parser.add_argument("--face-size", type=int, default=96,
                       help="Output face crop size in pixels (default: 96)")
    parser.add_argument("--padding", type=float, default=0.20,
                       help="Padding ratio around face (default: 0.20)")
    parser.add_argument("--confidence", type=float, default=0.6,
                       help="Minimum detection confidence (default: 0.6)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        extractor = FaceExtractor(
            face_size=args.face_size,
            padding_ratio=args.padding,
            min_confidence=args.confidence
        )
        
        saved, skipped = extractor.extract(args.video, args.label, args.fps)
        
        if saved == 0:
            logger.warning("No faces were extracted! Check if:")
            logger.warning("  1. Video contains visible faces")
            logger.warning("  2. Lighting conditions are adequate")
            logger.warning("  3. Try lowering confidence threshold (--confidence 0.3)")
            sys.exit(1)
        
        logger.info(f"✓ Successfully extracted {saved} faces")
        
    except KeyboardInterrupt:
        logger.info("\nExtraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=args.debug)
        sys.exit(1)

if __name__ == "__main__":
    main()