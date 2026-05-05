from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/ball_detection",
    name="train",
)

print("Training done. Best weights saved to runs/ball_detection/train/weights/best.pt")
