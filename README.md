# Table Tennis Analyzer

A computer vision pipeline that analyzes a table tennis match from a broadcast video. It tracks both players and the ball, counts rallies, reads the scoreboard, and turns the whole match into charts and highlight clips.

## What it does

- **Player tracking:** a YOLOv8 pose model finds both players in every frame. Their positions are saved to CSV and turned into a court-position heatmap for each player.
- **Ball and table detection:** a YOLOv8 segmentation model that I fine-tuned on a custom-labeled dataset with two classes, `table` and `ball` (`train_ball_model.py`, 50 epochs at 640px).
- **Real-world scale:** the detected table is used to convert pixels to meters (a standard table is 2.74 m long), so ball speed is reported in m/s.
- **Rally counter:** a shot is counted each time the ball changes direction after traveling a minimum distance, and a rally ends once the ball has been missing for 60 frames. The longest rally is exported as its own clip.
- **Scoreboard reading:** EasyOCR reads the on-screen score every 50 frames to chart how the score develops over the match.
- **Broadcast filtering:** frames are kept only when they show the wide gameplay camera (two players detected, far apart, neither shown in close-up), so replays and close-ups don't distort the statistics.

## Output

| File | Contents |
| --- | --- |
| `output_with_detections.mp4` | Annotated video with player boxes, the ball trail, ball speed, and a live rally counter |
| `longest_rally.mp4` | Clip of the longest rally in the match |
| `player_position_heatmap.png` | Where each player spent the match |
| `rally_chart.png` | Shots per rally, with the average and the longest rally |
| `score_chart.png` | Score progression read from the scoreboard |
| `ball_speed_chart.png` | Ball speed over time |
| `player_positions.csv`, `ball_positions.csv` | Raw per-frame positions |

## Tech stack

Python, Ultralytics YOLOv8 (pose and segmentation), OpenCV, EasyOCR, NumPy, Matplotlib

## Running it

```bash
pip install -r requirements.txt
python table_tennis_analyzer.py path/to/match.mp4
```

The fine-tuned ball model (`best.pt`) and the dataset images aren't included because of their size. The label files are in `Assignment2_Dataset/`. To train your own model, add the images next to the labels and run:

```bash
python train_ball_model.py
```

EasyOCR runs on the GPU by default (`gpu=True` in `table_tennis_analyzer.py`). Set it to `False` if you don't have a CUDA GPU.
