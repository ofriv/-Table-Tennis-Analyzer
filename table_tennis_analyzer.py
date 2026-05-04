import sys
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO

START_FRAME = 1200
END_FRAME = 3000
OCR_INTERVAL = 50  # read score every N valid frames

def main():
    model = YOLO("yolov8n-pose.pt")
    ocr_reader = easyocr.Reader(['en'], gpu=False)

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
    scores = []  # (frame_idx, p1_calc_score, p2_calc_score)
    background_frame = None
    valid_frame_count = 0

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
            all_boxes = [box.xyxy[0].tolist() for box in results[0].boxes]

            if not is_valid_frame(all_boxes, width, height):
                continue

            out.write(results[0].plot())

            people = []
            for x1, y1, x2, y2 in all_boxes:
                area = (x2 - x1) * (y2 - y1)
                people.append((area, (x1 + x2) / 2, (y1 + y2) / 2))

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

            if valid_frame_count % OCR_INTERVAL == 0:
                score = read_score(frame, width, height, ocr_reader)
                if score is not None:
                    print(f"Frame {frame_idx + 1}: p1={score[0]} p2={score[1]}")
                    scores.append((frame_idx + 1, score[0], score[1]))

            valid_frame_count += 1
            print(f"Frame {frame_idx + 1}/{end}", end="\r")

    cap.release()
    out.release()
    print("\nSaved output_with_detections.mp4 and player_positions.csv")

    create_heatmap(p1_positions, p2_positions, background_frame, width, height)
    create_score_chart(scores)


def is_valid_frame(boxes, width, height):
    if len(boxes) < 2:
        return False
    if any((y2 - y1) > 0.45 * height for x1, y1, x2, y2 in boxes):
        return False
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    top2 = sorted(zip(areas, boxes), reverse=True)[:2]
    cx1 = (top2[0][1][0] + top2[0][1][2]) / 2
    cx2 = (top2[1][1][0] + top2[1][1][2]) / 2
    if abs(cx1 - cx2) < 0.25 * width:
        return False
    return True


def read_score(frame, width, height, reader):
    crop = frame[int(height * 0.80):height, 0:int(width * 0.30)]
    results = reader.readtext(crop, allowlist='0123456789')

    numbers = []
    for bbox, text, conf in results:
        # scores are at most 2 digits and within realistic table tennis range
        if text.isdigit() and conf > 0.4 and len(text) <= 2 and int(text) <= 30:
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            numbers.append((cy, cx, int(text)))

    if len(numbers) < 4:
        return None

    numbers.sort(key=lambda n: n[0])
    mid_y = (numbers[0][0] + numbers[-1][0]) / 2
    row1 = sorted([n for n in numbers if n[0] <= mid_y], key=lambda n: n[1])
    row2 = sorted([n for n in numbers if n[0] > mid_y], key=lambda n: n[1])

    if len(row1) < 2 or len(row2) < 2:
        return None

    # two rightmost numbers per row: games won then current points
    p1 = row1[-2][2] * 10 + row1[-1][2]
    p2 = row2[-2][2] * 10 + row2[-1][2]
    return p1, p2


def create_score_chart(scores):
    frames   = [s[0] for s in scores]
    p1_scores = [s[1] for s in scores]
    p2_scores = [s[2] for s in scores]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frames, p1_scores, label="Player 1", color="blue", marker="o", markersize=3)
    ax.plot(frames, p2_scores, label="Player 2", color="red",  marker="o", markersize=3)
    ax.set_xlabel(f"Frame (sampled every {OCR_INTERVAL} valid frames)")
    ax.set_ylabel("Score (games won × 10 + current points)")
    ax.set_title("Player Scores Over Time")
    ax.legend()

    plt.tight_layout()
    plt.savefig("score_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved score_chart.png")


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
