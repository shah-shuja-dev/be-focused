import cv2, os, time, argparse

def record(label: str, duration: int = 120, out_dir: str = r"data\raw"):
    os.makedirs(out_dir, exist_ok=True)

   
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Try index 1 or 2 if 0 fails.")
        return

    ts     = int(time.time())
    path   = os.path.join(out_dir, f"{label}_{ts}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))

    print(f"Recording '{label}' for {duration}s  ->  {path}")
    print("Press Q in the window to stop early.")
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — stopping.")
            break
        writer.write(frame)
        elapsed = int(time.time() - start)
        cv2.putText(frame, f"{label}  {elapsed}s / {duration}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Recording (Q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",    required=True, choices=["focused", "not_focused"])
    ap.add_argument("--duration", type=int, default=120)
    args = ap.parse_args()
    record(args.label, args.duration)