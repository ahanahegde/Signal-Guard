
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

MODEL_PATH = Path("face_landmarker.task")

if not MODEL_PATH.exists():
    raise SystemExit("ERROR: face_landmarker.task not found in project folder.")

video_path = input("Enter video path: ").strip().strip('"')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise SystemExit("ERROR: Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

total_frames = 0
face_frames = 0

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
Options = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = Options(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH.resolve())),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp_ms = round(total_frames * 1000 / fps)
        total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            face_frames += 1
            landmarks = result.face_landmarks[0]
            h, w = frame.shape[:2]

            # Approximate forehead and cheek landmark locations
            forehead_ids = [10, 67, 69, 104, 108, 297, 299, 333, 337]
            cheek_ids = [50, 101, 118, 123, 205, 280, 330, 347, 425]

            for idx in forehead_ids:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            for idx in cheek_ids:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            label = "Face detected"
            color = (0, 255, 0)
        else:
            label = "No face detected"
            color = (0, 0, 255)

        cv2.putText(
            frame, label, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
        cv2.imshow("SignalGuard - Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print(f"\nTotal frames processed: {total_frames}")
print(f"Frames with face detected: {face_frames}")

if total_frames:
    print(f"Face detection rate: {face_frames / total_frames * 100:.2f}%")
