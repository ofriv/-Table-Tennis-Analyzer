import sys
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO

BALL_MODEL_PATH = "runs/segment/runs/ball_detection/train/weights/best.pt"

START_FRAME          = 1200
END_FRAME            = 3000
OCR_INTERVAL         = 50
BALL_TRAIL_LEN       = 30
DIRECTION_WINDOW     = 7   # frames to smooth ball direction
MIN_RALLY_SPEED      = 3.0  # pixels/frame — below this the ball is considered still
BALL_MISSING_TIMEOUT = 60  # frames without ball detection before rally ends


class RallyTracker:
    def __init__(self):
        self.active      = False
        self.shots       = 0
        self.missing     = 0
        self.recent_x    = []
        self.prev_dir    = 0  # -1 or 1
        self.start_frame = None
        self.rallies     = []  # (start_frame, end_frame, shot_count)

    def update(self, ball_pos, frame_idx, speed):
        if ball_pos is None:
            self.missing += 1
            if self.missing >= BALL_MISSING_TIMEOUT and self.active:
                self._end(frame_idx)
            return

        self.missing = 0
        bx, _ = ball_pos
        self.recent_x.append(bx)
        if len(self.recent_x) > DIRECTION_WINDOW:
            self.recent_x.pop(0)

        if len(self.recent_x) < DIRECTION_WINDOW or speed < MIN_RALLY_SPEED:
            return

        curr_dir = 1 if self.recent_x[-1] > self.recent_x[0] else -1
        if self.prev_dir != 0 and curr_dir != self.prev_dir:
            if not self.active:
                self.active      = True
                self.start_frame = frame_idx
                self.shots       = 1
            self.shots += 1
        self.prev_dir = curr_dir

    def _end(self, frame_idx):
        if self.shots >= 2:
            self.rallies.append((self.start_frame, frame_idx, self.shots))
        self.active      = False
        self.shots       = 0
        self.recent_x    = []
        self.prev_dir    = 0
        self.start_frame = None


def main():
    model      = YOLO("yolov8n-pose.pt")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    ball_model = YOLO(BALL_MODEL_PATH)

    cap          = cv2.VideoCapture(sys.argv[1])
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
    scores       = []
    ball_trail   = []
    ball_data    = []
    rally        = RallyTracker()
    background_frame  = None
    valid_frame_count = 0

    with open("player_positions.csv", "w", newline="") as pfile, \
         open("ball_positions.csv",   "w", newline="") as bfile:

        pwriter = csv.writer(pfile)
        pwriter.writerow(["frame", "player1_x", "player1_y", "player2_x", "player2_y"])

        bwriter = csv.writer(bfile)
        bwriter.writerow(["frame", "ball_x", "ball_y", "speed_px_per_frame"])

        for frame_idx in range(START_FRAME, end):
            ret, frame = cap.read()
            if not ret:
                break

            if background_frame is None:
                background_frame = frame.copy()

            results   = model(frame, verbose=False)
            all_boxes = [box.xyxy[0].tolist() for box in results[0].boxes]

            if not is_valid_frame(all_boxes, width, height):
                continue

            # --- ball detection ---
            ball_pos = detect_ball(frame, ball_model)
            speed = 0.0
            if ball_pos is not None:
                bx, by = ball_pos
                if ball_trail:
                    prev_frame, px, py, _ = ball_trail[-1]
                    frame_diff = max(frame_idx - prev_frame, 1)
                    speed = np.sqrt((bx - px)**2 + (by - py)**2) / frame_diff
                entry = (frame_idx, bx, by, speed)
                ball_trail.append(entry)
                ball_data.append(entry)
                if len(ball_trail) > BALL_TRAIL_LEN:
                    ball_trail.pop(0)
                bwriter.writerow([frame_idx + 1, f"{bx:.1f}", f"{by:.1f}", f"{speed:.2f}"])

            rally.update(ball_pos, frame_idx, speed)

            annotated = results[0].plot()
            annotated = draw_ball_trail(annotated, ball_trail, fps)
            annotated = draw_rally_overlay(annotated, rally)
            out.write(annotated)

            # --- player tracking ---
            people = []
            for x1, y1, x2, y2 in all_boxes:
                area = (x2 - x1) * (y2 - y1)
                people.append((area, (x1 + x2) / 2, (y1 + y2) / 2))

            people.sort(key=lambda p: p[0], reverse=True)
            people = people[:2]
            people.sort(key=lambda p: p[1])

            p1x = p1y = p2x = p2y = ""
            if len(people) >= 1:
                _, p1x, p1y = people[0]
                p1_positions.append((p1x, p1y))
            if len(people) >= 2:
                _, p2x, p2y = people[1]
                p2_positions.append((p2x, p2y))

            pwriter.writerow([frame_idx + 1, p1x, p1y, p2x, p2y])

            if valid_frame_count % OCR_INTERVAL == 0:
                score = read_score(frame, width, height, ocr_reader)
                if score is not None:
                    scores.append((frame_idx + 1, score[0], score[1]))

            valid_frame_count += 1
            print(f"Frame {frame_idx + 1}/{end}", end="\r")

    cap.release()
    out.release()
    print("\nSaved output_with_detections.mp4, player_positions.csv, ball_positions.csv")

    create_heatmap(p1_positions, p2_positions, background_frame, width, height)
    create_score_chart(scores)
    create_ball_speed_chart(ball_data, fps)
    create_rally_chart(rally.rallies)


def detect_ball(frame, model):
    results = model(frame, verbose=False)
    # class 1 = ball, class 0 = table
    ball_boxes = [box for box in results[0].boxes if int(box.cls[0]) == 1]
    if not ball_boxes:
        return None
    best = max(ball_boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = best.xyxy[0].tolist()
    return (x1 + x2) / 2, (y1 + y2) / 2


def draw_ball_trail(frame, trail, fps):
    if not trail:
        return frame

    speeds   = [t[3] for t in trail if t[3] > 0]
    max_spd  = max(speeds) if speeds else 1.0

    for i, (_, x, y, speed) in enumerate(trail):
        alpha     = (i + 1) / len(trail)
        spd_norm  = min(speed / max_spd, 1.0)
        # green (slow) → yellow → red (fast)
        r = int(255 * spd_norm)
        g = int(255 * (1 - spd_norm))
        cv2.circle(frame, (int(x), int(y)), max(3, int(8 * alpha)), (0, g, r), -1)

    # current speed label
    last_spd_pxs = trail[-1][3] * fps  # pixels/second
    cv2.putText(frame, f"Ball: {last_spd_pxs:.0f} px/s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame


def draw_rally_overlay(frame, rally):
    if rally.active:
        cv2.putText(frame, f"RALLY: {rally.shots} shots",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return frame


def create_rally_chart(rallies):
    if not rallies:
        print("No rallies detected")
        return

    shot_counts   = [r[2] for r in rallies]
    start_frames  = [r[0] for r in rallies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    max_shots = max(shot_counts)
    ax1.hist(shot_counts, bins=range(2, max_shots + 2), color="steelblue",
             edgecolor="white", align="left")
    ax1.set_xlabel("Shots per Rally")
    ax1.set_ylabel("Number of Rallies")
    ax1.set_title("Rally Length Distribution")
    ax1.set_xticks(range(2, max_shots + 1))

    ax2.scatter(start_frames, shot_counts, color="steelblue", s=30, alpha=0.7)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Shots in Rally")
    ax2.set_title("Rally Timeline")

    avg = sum(shot_counts) / len(shot_counts)
    fig.suptitle(f"Rally Analysis — {len(rallies)} rallies detected, avg {avg:.1f} shots")

    plt.tight_layout()
    plt.savefig("rally_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved rally_chart.png ({len(rallies)} rallies, avg {avg:.1f} shots)")


def is_valid_frame(boxes, width, height):
    if len(boxes) < 2:
        return False
    if any((y2 - y1) > 0.45 * height for x1, y1, x2, y2 in boxes):
        return False
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    top2  = sorted(zip(areas, boxes), reverse=True)[:2]
    cx1   = (top2[0][1][0] + top2[0][1][2]) / 2
    cx2   = (top2[1][1][0] + top2[1][1][2]) / 2
    if abs(cx1 - cx2) < 0.25 * width:
        return False
    return True


def read_score(frame, width, height, reader):
    crop    = frame[int(height * 0.80):height, 0:int(width * 0.30)]
    results = reader.readtext(crop, allowlist='0123456789')

    numbers = []
    for bbox, text, conf in results:
        if text.isdigit() and conf > 0.4 and len(text) <= 2 and int(text) <= 30:
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            numbers.append((cy, cx, int(text)))

    if len(numbers) < 4:
        return None

    numbers.sort(key=lambda n: n[0])
    mid_y = (numbers[0][0] + numbers[-1][0]) / 2
    row1  = sorted([n for n in numbers if n[0] <= mid_y], key=lambda n: n[1])
    row2  = sorted([n for n in numbers if n[0] >  mid_y], key=lambda n: n[1])

    if len(row1) < 2 or len(row2) < 2:
        return None

    p1 = row1[-2][2] * 10 + row1[-1][2]
    p2 = row2[-2][2] * 10 + row2[-1][2]
    return p1, p2


def create_score_chart(scores):
    frames    = [s[0] for s in scores]
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


def create_ball_speed_chart(ball_data, fps):
    if len(ball_data) < 2:
        print("Not enough ball detections for speed chart")
        return

    frames = [d[0] for d in ball_data]
    speeds = [d[3] * fps for d in ball_data]  # pixels/second

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frames, speeds, color="orange", linewidth=0.8, alpha=0.8)
    ax.fill_between(frames, speeds, alpha=0.2, color="orange")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Ball Speed (pixels / second)")
    ax.set_title("Table Tennis Ball Speed Over Time")

    plt.tight_layout()
    plt.savefig("ball_speed_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ball_speed_chart.png")


def make_heat_layer(positions, width, height):
    layer = np.zeros((height, width), dtype=np.float32)
    for x, y in positions:
        xi, yi = int(round(x)), int(round(y))
        layer[yi, xi] += 1.0
    sigma = width // 20
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return (layer / layer.sum()) * 100


def create_heatmap(p1_positions, p2_positions, background_frame, width, height):
    z1   = make_heat_layer(p1_positions, width, height)
    z2   = make_heat_layer(p2_positions, width, height)
    vmax = max(z1.max(), z2.max())

    cmap = plt.cm.jet
    bg   = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB).astype(np.float32)

    for z in [z1, z2]:
        z_norm  = np.clip(z / vmax, 0, 1)
        colored = (cmap(z_norm)[..., :3] * 255).astype(np.float32)
        alpha   = np.where(z_norm < 0.05, 0.0, np.clip(z_norm * 1.5, 0, 0.7))[..., np.newaxis]
        bg      = bg * (1 - alpha) + colored * alpha

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
