"""
SignalGuard temporal/spatial consistency diagnostics.

Usage:
  python temporal_consistency.py "path/to/video.mp4"

This module reports heuristic anomaly indicators, NOT a calibrated deepfake
probability. Motion, blur, pose changes, occlusion and compression can trigger
these indicators in genuine footage.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path("face_landmarker.task")
CSV_OUT = Path("temporal_consistency.csv")
JSON_OUT = Path("temporal_consistency.json")

# MediaPipe Face Mesh landmark groups.
KEY_IDS = [1, 10, 33, 61, 152, 199, 263, 291, 362, 234, 454]
LEFT_EYE = [33, 159, 145]
RIGHT_EYE = [362, 386, 374]
MOUTH = [61, 13, 14, 291]
POSE_IDS = [1, 152, 33, 263, 61, 291]
# Generic 3-D face points corresponding to nose, chin, eye corners and mouth
# corners. These provide rough relative pose estimates, not calibrated angles.


def estimate_head_pose(landmarks, width, height):
    image_points = np.array([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in POSE_IDS
    ], dtype=np.float64)
    model_points = np.array([
        [0.0, 0.0, 0.0], [0.0, 330.0, -65.0],
        [-225.0, -170.0, -135.0], [225.0, -170.0, -135.0],
        [-150.0, 150.0, -125.0], [150.0, 150.0, -125.0]
    ], dtype=np.float64)
    focal = float(width)
    camera = np.array([
        [focal, 0, width / 2.0],
        [0, focal, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    ok, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return (np.nan, np.nan, np.nan)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(-rotation[2, 0], sy)
        roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = np.arctan2(-rotation[2, 0], sy)
        roll = 0.0
    return tuple(float(np.degrees(v)) for v in (yaw, pitch, roll))


def robust_z(values):
    x = np.asarray(values, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-8:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def point_array(landmarks, ids, width, height):
    return np.array([[landmarks[i].x * width, landmarks[i].y * height]
                     for i in ids], dtype=float)


def norm_distance(a, b, scale):
    return float(np.linalg.norm(a - b) / max(scale, 1e-6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Video path")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    video_path = Path(args.video.strip('"')).resolve()
    model_path = Path(args.model)
    if not video_path.is_file():
        raise SystemExit(f"ERROR: Video not found: {video_path}")
    if not model_path.is_file():
        raise SystemExit(f"ERROR: Face landmark model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    base = python.BaseOptions(model_asset_path=str(model_path.resolve()))
    options = vision.FaceLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO,
        num_faces=1, output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    rows = []
    prev_norm = None
    prev_gray = None
    prev_t = None
    frame_idx = 0
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            t = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(round(t * 1000)))
            row = {"frame": frame_idx, "time_seconds": t, "face_detected": 0}
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=float)
                xspan = max(np.ptp(coords[:, 0]), 1e-6)
                yspan = max(np.ptp(coords[:, 1]), 1e-6)
                scale = float(np.hypot(xspan * w, yspan * h))
                center = np.array([np.mean(coords[:, 0]), np.mean(coords[:, 1])])
                normalized = (coords[:, :2] - center) / np.array([xspan, yspan])
                row["face_detected"] = 1
                row["face_center_x"] = center[0]
                row["face_center_y"] = center[1]
                row["face_width_norm"] = xspan
                row["face_height_norm"] = yspan
                row["landmark_step"] = (float(np.mean(np.linalg.norm(
                    normalized[KEY_IDS] - prev_norm[KEY_IDS], axis=1
                ))) if prev_norm is not None else np.nan)
                row["left_eye_aspect"] = (
                    norm_distance(point_array(lm, [159], w, h)[0],
                                  point_array(lm, [145], w, h)[0], scale)
                )
                row["right_eye_aspect"] = (
                    norm_distance(point_array(lm, [386], w, h)[0],
                                  point_array(lm, [374], w, h)[0], scale)
                )
                row["mouth_open_ratio"] = (
                    norm_distance(point_array(lm, [13], w, h)[0],
                                  point_array(lm, [14], w, h)[0], scale)
                )
                row["mouth_width_ratio"] = (
                    norm_distance(point_array(lm, [61], w, h)[0],
                                  point_array(lm, [291], w, h)[0], scale)
                )
                row["nose_depth_proxy"] = float(coords[1, 2])
                yaw, pitch, roll = estimate_head_pose(lm, w, h)
                row["head_yaw_deg"] = yaw
                row["head_pitch_deg"] = pitch
                row["head_roll_deg"] = roll
                row["pose_matrix_available"] = int(bool(
                    result.facial_transformation_matrixes
                ))
                row["blendshape_count"] = (
                    len(result.face_blendshapes[0])
                    if result.face_blendshapes else 0
                )
                prev_norm = normalized.copy()
            else:
                row.update({
                    "landmark_step": np.nan, "left_eye_aspect": np.nan,
                    "right_eye_aspect": np.nan, "mouth_open_ratio": np.nan,
                    "mouth_width_ratio": np.nan, "nose_depth_proxy": np.nan,
                    "head_yaw_deg": np.nan, "head_pitch_deg": np.nan,
                    "head_roll_deg": np.nan,
                    "pose_matrix_available": 0, "blendshape_count": 0,
                })
                prev_norm = None

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.linalg.norm(flow, axis=2)
                row["optical_flow_median"] = float(np.median(mag))
                row["optical_flow_p95"] = float(np.percentile(mag, 95))
                row["frame_difference"] = float(np.mean(cv2.absdiff(prev_gray, gray)))
            else:
                row["optical_flow_median"] = np.nan
                row["optical_flow_p95"] = np.nan
                row["frame_difference"] = np.nan

            row["laplacian_blur_score"] = float(
                cv2.Laplacian(gray, cv2.CV_64F).var()
            )
            rows.append(row)
            prev_gray = gray
            prev_t = t
            frame_idx += 1

    cap.release()
    if not rows:
        raise SystemExit("ERROR: No frames read from video.")

    df = pd.DataFrame(rows)
    df["landmark_step_robust_z"] = robust_z(df["landmark_step"].to_numpy())
    df["flow_robust_z"] = robust_z(df["optical_flow_median"].to_numpy())
    df["frame_diff_robust_z"] = robust_z(df["frame_difference"].to_numpy())
    df["mouth_change"] = df["mouth_open_ratio"].diff().abs()
    df["eye_asymmetry"] = (df["left_eye_aspect"] - df["right_eye_aspect"]).abs()
    df["geometry_jump_flag"] = (
        (df["landmark_step_robust_z"].abs() > 5) |
        (df["frame_diff_robust_z"].abs() > 5)
    ) & (df["face_detected"] == 1)

    face_rate = float(df["face_detected"].mean())
    anomaly_rate = float(df["geometry_jump_flag"].mean())
    # Score is a descriptive anomaly index, not P(fake).
    anomaly_index = float(np.clip(
        0.50 * min(anomaly_rate / 0.10, 1.0) +
        0.25 * min(float(np.nanmean(np.abs(df["landmark_step_robust_z"])) / 5), 1.0) +
        0.25 * min(float(np.nanmean(np.abs(df["frame_diff_robust_z"])) / 5), 1.0),
        0, 1
    ))
    summary = {
        "video": str(video_path),
        "frames": int(len(df)),
        "fps_metadata": fps,
        "face_detection_rate": face_rate,
        "geometry_jump_rate": anomaly_rate,
        "temporal_anomaly_index": anomaly_index,
        "interpretation": "heuristic anomaly index only; not a deepfake probability",
        "limitations": [
            "Normal motion, blur, occlusion and compression can trigger anomalies.",
            "Modern generators may maintain temporal consistency.",
            "Requires validation on labelled real and manipulated videos."
        ],
    }
    df.to_csv(CSV_OUT, index=False)
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SIGNALGUARD - TEMPORAL/SPATIAL CONSISTENCY")
    print(f"Frames: {len(df)} | Face detection rate: {face_rate*100:.1f}%")
    print(f"Geometry jump rate: {anomaly_rate*100:.2f}%")
    print(f"Temporal anomaly index (NOT P(fake)): {anomaly_index:.3f}")
    print(f"Saved: {CSV_OUT}, {JSON_OUT}")


if __name__ == "__main__":
    main()
