"""
SignalGuard image/video forensic diagnostics.

Usage:
  python video_forensics.py "path/to/video.mp4"

Uses interpretable frame-level proxies: high-frequency DCT energy, block-boundary
discontinuity, resampling periodicity, frame-to-frame change and face-boundary
edge discontinuity. These are NOT universal deepfake fingerprints and do not
produce a calibrated probability of manipulation.
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
CSV_OUT = Path("video_forensics.csv")
JSON_OUT = Path("video_forensics.json")


def robust_z(values):
    x = np.asarray(values, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-8:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def blockiness(gray):
    g = gray.astype(np.float32)
    h, w = g.shape
    vals = []
    for axis, dim in ((1, w), (0, h)):
        if dim < 16:
            continue
        diff = np.abs(np.diff(g, axis=axis))
        # Compare pixel differences at 8-pixel boundaries to nearby non-boundaries.
        boundary_idx = np.arange(7, dim - 1, 8)
        other_idx = np.array([i for i in range(diff.shape[axis]) if i % 8 != 7])
        if len(boundary_idx) == 0 or len(other_idx) == 0:
            continue
        if axis == 1:
            b = diff[:, boundary_idx]
            o = diff[:, other_idx]
        else:
            b = diff[boundary_idx, :]
            o = diff[other_idx, :]
        vals.append(float(np.mean(b) / (np.mean(o) + 1e-6)))
    return float(np.mean(vals)) if vals else np.nan


def high_frequency_energy(gray):
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    h, w = dct.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(xx * xx + yy * yy)
    total = float(np.sum(dct * dct) + 1e-9)
    high = float(np.sum((dct * dct)[radius > 45]))
    return high / total


def resampling_periodicity(gray):
    # Measure periodicity in mean absolute horizontal/vertical gradients.
    g = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = np.mean(np.abs(np.diff(g, axis=1)), axis=0)
    gy = np.mean(np.abs(np.diff(g, axis=0)), axis=1)
    def periodicity(v):
        v = v - np.mean(v)
        if len(v) < 24 or np.std(v) < 1e-6:
            return 0.0
        ac = np.correlate(v, v, mode="full")[len(v)-1:]
        ac = ac / (ac[0] + 1e-9)
        candidates = [k for k in range(2, min(17, len(ac)//2)) if k < len(ac)]
        return float(max((ac[k] for k in candidates), default=0.0))
    return float((periodicity(gx) + periodicity(gy)) / 2)


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
        num_faces=1
    )
    rows = []
    previous_gray = None
    frame_idx = 0
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = frame_idx / fps
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = landmarker.detect_for_video(mp_image, int(round(t * 1000)))
            row = {
                "frame": frame_idx, "time_seconds": t,
                "high_frequency_energy": high_frequency_energy(gray),
                "blockiness_ratio": blockiness(gray),
                "resampling_periodicity": resampling_periodicity(gray),
                "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "face_detected": int(bool(result.face_landmarks)),
            }
            if previous_gray is not None:
                row["frame_difference"] = float(cv2.absdiff(previous_gray, gray).mean())
            else:
                row["frame_difference"] = np.nan

            # Face boundary proxy: edge density near the face bbox perimeter
            # compared with a thin ring just outside it. This is only a weak cue.
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                h, w = gray.shape
                xs = np.array([p.x * w for p in lm])
                ys = np.array([p.y * h for p in lm])
                x1, x2 = int(max(0, xs.min())), int(min(w, xs.max()))
                y1, y2 = int(max(0, ys.min())), int(min(h, ys.max()))
                if x2 - x1 > 10 and y2 - y1 > 10:
                    edges = cv2.Canny(gray, 80, 160)
                    pad = max(2, int(0.025 * min(x2-x1, y2-y1)))
                    inner = edges[y1:y2, x1:x2]
                    ring_mask = np.zeros_like(inner, dtype=np.uint8)
                    ring_mask[:pad, :] = 1
                    ring_mask[-pad:, :] = 1
                    ring_mask[:, :pad] = 1
                    ring_mask[:, -pad:] = 1
                    row["face_boundary_edge_density"] = float(
                        inner[ring_mask > 0].mean() / 255.0
                    )
                else:
                    row["face_boundary_edge_density"] = np.nan
            else:
                row["face_boundary_edge_density"] = np.nan

            rows.append(row)
            previous_gray = gray
            frame_idx += 1
    cap.release()
    if not rows:
        raise SystemExit("ERROR: No frames read from video.")

    df = pd.DataFrame(rows)
    feature_cols = [
        "high_frequency_energy", "blockiness_ratio",
        "resampling_periodicity", "laplacian_variance",
        "frame_difference", "face_boundary_edge_density"
    ]
    for col in feature_cols:
        df[col + "_robust_z"] = robust_z(df[col].to_numpy())
    zcols = [c + "_robust_z" for c in feature_cols]
    df["forensic_deviation_index"] = np.clip(
        df[zcols].abs().mean(axis=1, skipna=True) / 5.0, 0, 1
    )
    summary = {
        "video": str(video_path),
        "frames": int(len(df)),
        "fps_metadata": fps,
        "mean_features": {
            c: (float(df[c].mean()) if df[c].notna().any() else None)
            for c in feature_cols
        },
        "forensic_deviation_index": float(df["forensic_deviation_index"].mean()),
        "interpretation": "heuristic forensic deviation index only; not P(fake)",
        "limitations": [
            "Compression and resizing can change these features.",
            "Natural content varies widely; anomaly does not imply manipulation.",
            "Modern generators may not exhibit these cues.",
            "Requires validation on labelled datasets and unseen manipulation methods."
        ],
    }
    df.to_csv(CSV_OUT, index=False)
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SIGNALGUARD - VIDEO FORENSICS")
    print(f"Frames: {len(df)}")
    print(f"Forensic deviation index (NOT P(fake)): {summary['forensic_deviation_index']:.3f}")
    for k, v in summary["mean_features"].items():
        print(f"{k}: {v if v is not None else 'unavailable'}")
    print(f"Saved: {CSV_OUT}, {JSON_OUT}")


if __name__ == "__main__":
    main()
