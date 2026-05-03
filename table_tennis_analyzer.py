import sys
import cv2
from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("Usage: python table_tennis_analyzer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = YOLO("yolov8n-pose.pt")
    results = model(image_path)

    img = cv2.imread(image_path)

    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw pose keypoints and skeleton
        if keypoints is not None:
            for person_kps in keypoints.xy:
                for x, y in person_kps:
                    x, y = int(x), int(y)
                    if x > 0 and y > 0:
                        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    output_path = "output.jpg"
    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")

    max_display_width = 1280
    h, w = img.shape[:2]
    if w > max_display_width:
        scale = max_display_width / w
        display_img = cv2.resize(img, (max_display_width, int(h * scale)))
    else:
        display_img = img

    cv2.imshow("Pose Detection", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
