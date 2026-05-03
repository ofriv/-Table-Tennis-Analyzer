import sys
import cv2
from ultralytics import YOLO

START_FRAME = 0
END_FRAME = 200

def main():
    if len(sys.argv) < 2:
        print("Usage: python table_tennis_analyzer.py <video_path>")
        sys.exit(1)

    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(sys.argv[1])
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end = total_frames if END_FRAME == 0 else END_FRAME

    out = cv2.VideoWriter(
        "output_with_detections.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    for frame_idx in range(START_FRAME, end):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)
        print(f"Frame {frame_idx}/{end - 1}", end="\r")

    cap.release()
    out.release()
    print("\nSaved output_with_detections.mp4")

if __name__ == "__main__":
    main()
