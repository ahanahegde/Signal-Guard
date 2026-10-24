"""
SignalGuard - Baseline Deepfake Video Classifier.

Reference implementations from FaceForensics++ (XceptionNet: Rossler et al., ICCV 2019)
and Kaggle/Meta DFDC Winning Solution (EfficientNet-B4: Seferbekov et al., 2020).

Workflow:
  1. Reads video frames and samples evenly across duration.
  2. Detects and extracts normalized face crops via MediaPipe.
  3. Preprocesses crops (299x299, ImageNet standardization).
  4. Computes per-frame fake probabilities using deep CNN features.
  5. Aggregates frame probabilities into a robust video-level verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ============================================================
# 1. MODEL ARCHITECTURE (FaceForensics++ Xception / EfficientNet)
# ============================================================

def build_deepfake_backbone(model_name: str = "efficientnet", num_classes: int = 2) -> Any:
    """
    Builds the classifier backbone. Uses timm if available, or falls back to
    torchvision EfficientNet/ResNet with a custom binary classification head.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required to run deep learning backbones. "
            "Please install PyTorch via: pip install torch torchvision timm"
        )

    try:
        import timm
        # 'legacy_xception' is the FaceForensics++ architecture.
        # 'tf_efficientnet_b4' is the Kaggle DFDC winning architecture.
        backbone_id = "legacy_xception" if "xception" in model_name.lower() else "tf_efficientnet_b4"
        print(f"[*] Initializing '{backbone_id}' via timm...")
        model = timm.create_model(backbone_id, pretrained=True, num_classes=num_classes)
        return model
    except ImportError:
        import torchvision.models as models
        print(f"[*] 'timm' not found; using torchvision EfficientNet-B4...")
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model


# ============================================================
# 2. FACE DETECTION & PREPROCESSING PIPELINE
# ============================================================

class FacePreprocessor:
    """Extracts stabilized, margin-padded face crops using MediaPipe."""

    def __init__(self, target_size: int = 299, margin: float = 0.25):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is required for face detection: pip install mediapipe")

        self.target_size = target_size
        self.margin = margin
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )

        # Standard ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def extract_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(frame_rgb)

        if not results.detections:
            return None

        # Select detection with highest confidence
        best_detection = max(results.detections, key=lambda d: d.score[0])
        bbox = best_detection.location_data.relative_bounding_box

        # Convert relative to pixel coordinates
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

        # Add margin
        dx = int(box_w * self.margin)
        dy = int(box_h * self.margin)

        x1 = max(0, x_min - dx)
        y1 = max(0, y_min - dy)
        x2 = min(w, x_min + box_w + dx)
        y2 = min(h, y_min + box_h + dy)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        crop = frame_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(
            crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
        )

        # Normalize: [0, 255] -> standard normal distribution
        norm_crop = (crop_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        # Transpose HWC -> CHW for PyTorch tensor
        return np.transpose(norm_crop, (2, 0, 1))

    def close(self):
        self.mp_face.close()


# ============================================================
# 3. VIDEO INFERENCE & TEMPORAL AGGREGATION
# ============================================================

def classify_video(
    video_path: str,
    model: Any,
    device: Any,
    max_frames: int = 32,
    fake_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Samples frames across video duration, computes frame-level fake probabilities,
    and aggregates them into a video-level verdict.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required: pip install torch")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video at: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Select frame indices evenly spaced across the entire video
    sample_indices = np.linspace(
        0, total_frames - 1, min(max_frames, total_frames), dtype=int
    )
    preprocessor = FacePreprocessor(target_size=299)

    frame_probs: List[float] = []

    model.eval()
    with torch.no_grad():
        for idx in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                break

            if idx not in sample_indices:
                continue

            face_tensor = preprocessor.extract_face(frame)
            if face_tensor is None:
                continue

            tensor = torch.from_numpy(face_tensor).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            # Index 1 = P(Fake), Index 0 = P(Real)
            p_fake = (
                float(probs[1])
                if probs.shape[0] > 1
                else float(torch.sigmoid(logits).item())
            )
            frame_probs.append(p_fake)

    cap.release()
    preprocessor.close()

    if not frame_probs:
        return {
            "verdict": "ERROR",
            "reason": "No valid faces could be detected in the sampled frames.",
            "frames_analyzed": 0,
        }

    # Temporal aggregation:
    # 1. Mean probability across all sampled frames
    # 2. Median probability (robust against transient detector noise)
    # 3. Top-10% worst frames (catches sparse deepfakes where only brief intervals are altered)
    mean_prob = float(np.mean(frame_probs))
    median_prob = float(np.median(frame_probs))
    k_frames = max(1, len(frame_probs) // 10)
    top_k_prob = float(np.mean(sorted(frame_probs, reverse=True)[:k_frames]))

    video_fake_score = 0.60 * mean_prob + 0.40 * top_k_prob
    verdict = "DEEPFAKE" if video_fake_score >= fake_threshold else "REAL"

    return {
        "verdict": verdict,
        "fake_probability": round(video_fake_score, 4),
        "mean_frame_probability": round(mean_prob, 4),
        "median_frame_probability": round(median_prob, 4),
        "top_k_worst_frame_probability": round(top_k_prob, 4),
        "frames_analyzed": len(frame_probs),
        "fps": round(fps, 2),
        "duration_seconds": round(total_frames / fps, 2),
    }


# ============================================================
# 4. ENTRY POINT
# ============================================================

def main():
    if not TORCH_AVAILABLE:
        print(
            "ERROR: PyTorch is not installed in the active environment.\n"
            "To use deep learning backbones (Xception / EfficientNet), install dependencies:\n"
            "    pip install torch torchvision timm\n"
            "\nAlternatively, run the scikit-learn / OpenCV forensic baseline with:\n"
            "    python baseline_classifier.py --predict <video_path>"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Baseline Deepfake Video Classifier (FaceForensics++/DFDC)"
    )
    parser.add_argument("video_path", type=str, help="Path to input video file (.mp4)")
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet",
        choices=["xception", "efficientnet"],
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to custom fine-tuned weights (.pth)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="Number of frames to sample across the video",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Fake decision threshold (0.0 - 1.0)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    model = build_deepfake_backbone(args.backbone, num_classes=2).to(device)
    if args.weights and Path(args.weights).is_file():
        print(f"[*] Loading custom checkpoint: {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))

    print(f"[*] Analyzing video: {args.video_path}")
    results = classify_video(
        video_path=args.video_path,
        model=model,
        device=device,
        max_frames=args.max_frames,
        fake_threshold=args.threshold,
    )

    print("\n" + "=" * 52)
    print("           BASELINE CLASSIFICATION RESULT           ")
    print("=" * 52)
    print(f"Verdict:              {results['verdict']}")
    print(f"Overall Fake Score:   {results.get('fake_probability', 0.0) * 100:.2f}%")
    print(f"Mean Frame Score:     {results.get('mean_frame_probability', 0.0) * 100:.2f}%")
    print(f"Median Frame Score:   {results.get('median_frame_probability', 0.0) * 100:.2f}%")
    print(f"Top-K Peak Fake Score:{results.get('top_k_worst_frame_probability', 0.0) * 100:.2f}%")
    print(f"Frames Sampled:       {results['frames_analyzed']}")
    print("=" * 52)


if __name__ == "__main__":
    main()
