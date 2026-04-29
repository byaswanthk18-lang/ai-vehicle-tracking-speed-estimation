import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import os

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"

if not os.path.exists(INPUT_VIDEO):
    print(f"ERROR: {INPUT_VIDEO} not found in this folder")
    exit()

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("ERROR: input.mp4 is not opening. Check video path or format.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if width == 0 or height == 0:
    print("ERROR: Could not read video width/height.")
    exit()

if fps == 0 or fps is None:
    fps = 30

print("Video loaded successfully")
print("Width:", width)
print("Height:", height)
print("FPS:", fps)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: VideoWriter failed to open.")
    exit()

tracker = DeepSort(max_age=30)
track_history = defaultdict(list)

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

line_y = int(height * 0.60)

entering_counts = defaultdict(int)
leaving_counts = defaultdict(int)

counted_entering = set()
counted_leaving = set()

SPEED_SCALE = 0.45
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model(frame, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                w = x2 - x1
                h = y2 - y1

                class_name = vehicle_classes[cls]
                detections.append(([x1, y1, w, h], conf, class_name))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        class_name = track.get_det_class() or "vehicle"

        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        track_history[track_id].append((cx, cy))

        if len(track_history[track_id]) > 20:
            track_history[track_id].pop(0)

        speed_kmh = 0

        if len(track_history[track_id]) >= 2:
            x_prev, y_prev = track_history[track_id][-2]
            x_curr, y_curr = track_history[track_id][-1]

            pixel_distance = np.sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2)
            speed_kmh = pixel_distance * fps * SPEED_SCALE

            if y_prev < line_y <= y_curr and track_id not in counted_entering:
                entering_counts[class_name] += 1
                counted_entering.add(track_id)

            elif y_prev > line_y >= y_curr and track_id not in counted_leaving:
                leaving_counts[class_name] += 1
                counted_leaving.add(track_id)

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

        label = f"{track_id}:{class_name} {speed_kmh:.0f} km/h"
        cv2.putText(frame, label, (l, max(t - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        points = track_history[track_id]
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (255, 0, 255), 2)

    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)

    cv2.rectangle(frame, (20, 20), (430, 130), (80, 80, 255), -1)
    cv2.putText(frame, "Numbers of Vehicles Leaving", (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    y_pos = 90
    for name, value in leaving_counts.items():
        cv2.putText(frame, f"{name}: {value}", (35, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30

    cv2.rectangle(frame, (width - 450, 20), (width - 20, 160), (80, 80, 255), -1)
    cv2.putText(frame, "Number of Vehicles Entering", (width - 430, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    y_pos = 90
    for name, value in entering_counts.items():
        cv2.putText(frame, f"{name}: {value}", (width - 430, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30

    out.write(frame)
    cv2.imshow("AI Vehicle Tracking Output", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done. Frames processed: {frame_count}")
print("Output saved as output.mp4")