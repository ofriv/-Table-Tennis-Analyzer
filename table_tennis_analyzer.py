import sys
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

START_FRAME = 0
END_FRAME = 2000

def main():
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(sys.argv[1])
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end = total_frames if END_FRAME == 0 else END_FRAME

    out = cv2.VideoWriter(
        "output_with_detections.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    p1_positions = []
    p2_positions = []
    background_frame = None

    with open("player_positions.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "player1_position_x", "player1_position_y",
                         "player2_position_x", "player2_position_y"])

        for frame_idx in range(START_FRAME, end):
            ret, frame = cap.read()
            if not ret:
                break

            if background_frame is None:
                background_frame = frame.copy()

            results = model(frame, verbose=False)
            out.write(results[0].plot())

            people = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                people.append((area, (x1 + x2) / 2, (y1 + y2) / 2))

            # Keep the 2 largest boxes (players are closer to camera than referee)
            people.sort(key=lambda p: p[0], reverse=True)
            people = people[:2]
            people.sort(key=lambda p: p[1])  # left to right → player1, player2

            p1x = p1y = p2x = p2y = ""
            if len(people) >= 1:
                _, p1x, p1y = people[0]
                p1_positions.append((p1x, p1y))
            if len(people) >= 2:
                _, p2x, p2y = people[1]
                p2_positions.append((p2x, p2y))

            writer.writerow([frame_idx + 1, p1x, p1y, p2x, p2y])
            print(f"Frame {frame_idx + 1}/{end}", end="\r")

    cap.release()
    out.release()
    print("\nSaved output_with_detections.mp4 and player_positions.csv")

    create_heatmap(p1_positions, p2_positions, background_frame, width, height)


def make_heat_layer(positions, width, height):
    layer = np.zeros((height, width), dtype=np.float32)
    for x, y in positions:
        xi, yi = int(round(x)), int(round(y))
        layer[yi, xi] += 1.0
    sigma = width // 20
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return (layer / layer.sum()) * 100


def create_heatmap(p1_positions, p2_positions, background_frame, width, height):
    z1 = make_heat_layer(p1_positions, width, height)
    z2 = make_heat_layer(p2_positions, width, height)
    vmax = max(z1.max(), z2.max())

    cmap = plt.cm.jet
    bg = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB).astype(np.float32)

    for z in [z1, z2]:
        z_norm = np.clip(z / vmax, 0, 1)
        colored = (cmap(z_norm)[..., :3] * 255).astype(np.float32)
        alpha = np.where(z_norm < 0.05, 0.0, np.clip(z_norm * 1.5, 0, 0.7))[..., np.newaxis]
        bg = bg * (1 - alpha) + colored * alpha

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(bg.astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Combined Player Position Heatmap")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="% of Frames")

    plt.tight_layout()
    plt.savefig("player_position_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved player_position_heatmap.png")


if __name__ == "__main__":
    main()
