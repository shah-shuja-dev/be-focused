import cv2, os, argparse
import mediapipe as mp

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def extract(video_path: str, label: str, fps_sample: int = 1):
    out_dir = os.path.join("data", "processed", label)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 20
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detector.process(rgb)

        if not result.detections:
            skipped += 1
            continue

        det = result.detections[0]
        bb = det.location_data.relative_bounding_box
        H, W = frame.shape[:2]

        x1 = max(0, int((bb.xmin - 0.20 * bb.width) * W))
        y1 = max(0, int((bb.ymin - 0.20 * bb.height) * H))
        x2 = min(W, int((bb.xmin + 1.20 * bb.width) * W))
        y2 = min(H, int((bb.ymin + 1.20 * bb.height) * H))

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (96, 96))
        fname = os.path.join(out_dir, f"{label}_{base}_{frame_idx:05d}.jpg")
        cv2.imwrite(fname, face)
        saved += 1

    cap.release()
    print(f"  {os.path.basename(video_path)}: saved={saved}  skipped(no face)={skipped}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--label", required=True, choices=["focused", "not_focused"])
    ap.add_argument("--fps", type=int, default=1)
    args = ap.parse_args()
    extract(args.video, args.label, args.fps)