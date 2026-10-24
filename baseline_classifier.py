"""
SignalGuard Baseline Deepfake Classifier.

Implements a modular, reproducible baseline classifier for genuine vs. manipulated
face videos. Extracts spatial, frequency-domain (azimuthal FFT residual), and forensic
boundary representations, trained via L2-regularized logistic regression with Platt scaling
probability calibration.

Operates independently of SignalGuard's physiological eligibility gate to enable
rigorous comparison between un-gated vs. gated deepfake detection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset_loader import (
    LABEL_GENUINE,
    LABEL_MANIPULATED,
    NAME_GENUINE,
    NAME_MANIPULATED,
    LABEL_TO_NAME,
    DatasetLoader,
    VideoRecord,
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "baseline_classifier.joblib"
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_DIR / "baseline_config.json"

FEATURE_NAMES = [
    "high_freq_ratio",
    "azimuthal_spectral_variance",
    "laplacian_variance",
    "edge_gradient_density",
    "inter_frame_difference",
    "color_channel_discrepancy",
    "spectral_falloff_slope",
    "temporal_gradient_variance",
]


def extract_frame_forensic_features(frame: np.ndarray) -> dict[str, float]:
    """
    Extract spatial, boundary, and 2D-FFT frequency metrics from a single frame.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    # 1. 2D FFT Spectral Analysis
    # Generative models / upsampling operators exhibit high-frequency residual anomalies (Durall et al., Frank et al.)
    f_transform = np.fft.fft2(gray.astype(float))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)

    # Compute radial distance grid from center
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    r_grid = np.hypot(x_idx - cx, y_idx - cy)
    max_radius = np.hypot(cx, cy)

    # High-frequency ratio (energy in top 30% frequency band / total energy)
    high_freq_mask = r_grid >= 0.70 * max_radius
    total_energy = float(np.sum(magnitude_spectrum))
    high_freq_energy = float(np.sum(magnitude_spectrum[high_freq_mask]))
    high_freq_ratio = high_freq_energy / max(total_energy, 1e-10)

    # Azimuthal spectral variance: energy distribution across 8 angular sectors
    theta_grid = np.arctan2(y_idx - cy, x_idx - cx)  # [-pi, pi]
    sector_energies = []
    for sector in range(8):
        angle_min = -np.pi + sector * (2.0 * np.pi / 8.0)
        angle_max = angle_min + (2.0 * np.pi / 8.0)
        sector_mask = (theta_grid >= angle_min) & (theta_grid < angle_max)
        sector_energies.append(float(np.mean(magnitude_spectrum[sector_mask])) if np.any(sector_mask) else 0.0)
    azimuthal_variance = float(np.var(sector_energies) / max(np.mean(sector_energies) ** 2, 1e-10))

    # Spectral falloff slope (ratio of low-to-mid energy vs mid-to-high energy)
    low_mask = (r_grid < 0.35 * max_radius) & (r_grid > 2)
    mid_high_mask = (r_grid >= 0.35 * max_radius)
    low_e = float(np.mean(magnitude_spectrum[low_mask])) if np.any(low_mask) else 1.0
    mid_high_e = float(np.mean(magnitude_spectrum[mid_high_mask])) if np.any(mid_high_mask) else 1.0
    spectral_falloff_slope = low_e / max(mid_high_e, 1e-10)

    # 2. Spatial Texture & Edge Gradients
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_gradient_density = float(np.mean(np.hypot(sobel_x, sobel_y)))

    # 3. Color channel discrepancy
    color_discrepancy = 0.0
    if frame.ndim == 3 and frame.shape[2] == 3:
        b, g, r = frame[:, :, 0].astype(float), frame[:, :, 1].astype(float), frame[:, :, 2].astype(float)
        # Difference between red/blue and green gradients
        diff_rg = np.mean(np.abs(r - g))
        diff_bg = np.mean(np.abs(b - g))
        color_discrepancy = float((diff_rg + diff_bg) / 2.0)

    return {
        "high_freq_ratio": high_freq_ratio,
        "azimuthal_spectral_variance": azimuthal_variance,
        "laplacian_variance": laplacian_var,
        "edge_gradient_density": edge_gradient_density,
        "color_channel_discrepancy": color_discrepancy,
        "spectral_falloff_slope": spectral_falloff_slope,
    }


def extract_video_features(
    video_path: str | Path,
    max_frames: int = 60,
    stride: int = 2,
) -> tuple[np.ndarray | None, dict[str, float] | None, str | None]:
    """
    Extract aggregate multi-frame spatial, frequency, and temporal forensic features from a video.
    Returns: (feature_vector, feature_dict, error_string)
    """
    path = Path(video_path)
    if not path.is_file():
        return None, None, f"Video file not found: {path}"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, f"OpenCV could not open video: {path}"

    frame_features = []
    prev_gray = None
    frame_diffs = []
    frame_idx = 0
    read_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                h, w = frame.shape[:2]
                if h < 16 or w < 16:
                    continue
                feat = extract_frame_forensic_features(frame)
                frame_features.append(feat)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    diff = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))))
                    frame_diffs.append(diff)
                prev_gray = gray
                read_count += 1
                if read_count >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    if not frame_features:
        return None, None, "No readable frames could be processed from video."

    # Compute video-level temporal aggregations
    avg_high_freq = float(np.median([f["high_freq_ratio"] for f in frame_features]))
    avg_azimuthal = float(np.median([f["azimuthal_spectral_variance"] for f in frame_features]))
    avg_laplacian = float(np.median([f["laplacian_variance"] for f in frame_features]))
    avg_edge_density = float(np.median([f["edge_gradient_density"] for f in frame_features]))
    avg_color_disc = float(np.median([f["color_channel_discrepancy"] for f in frame_features]))
    avg_slope = float(np.median([f["spectral_falloff_slope"] for f in frame_features]))

    avg_frame_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0
    temp_grad_var = float(np.var(frame_diffs)) if len(frame_diffs) > 1 else 0.0

    aggregated: dict[str, float] = {
        "high_freq_ratio": avg_high_freq,
        "azimuthal_spectral_variance": avg_azimuthal,
        "laplacian_variance": avg_laplacian,
        "edge_gradient_density": avg_edge_density,
        "inter_frame_difference": avg_frame_diff,
        "color_channel_discrepancy": avg_color_disc,
        "spectral_falloff_slope": avg_slope,
        "temporal_gradient_variance": temp_grad_var,
    }

    feature_vector = np.array([aggregated[name] for name in FEATURE_NAMES], dtype=float)
    return feature_vector, aggregated, None


class BaselineClassifier:
    """
    Baseline Real vs. Manipulated Video Classifier.
    Employs StandardScaler + LogisticRegression with CalibratedClassifierCV (Platt scaling).
    """

    def __init__(
        self,
        c_regularization: float = 1.0,
        decision_threshold: float = 0.5,
        random_state: int = 42,
    ):
        self.c_regularization = c_regularization
        self.decision_threshold = decision_threshold
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.calibrated_model: CalibratedClassifierCV | None = None
        self.feature_names = list(FEATURE_NAMES)
        self.training_metadata: dict[str, Any] = {}

    def fit(
        self,
        records: Sequence[VideoRecord],
        val_records: Sequence[VideoRecord] | None = None,
    ) -> dict[str, Any]:
        """
        Extract features and train the calibrated baseline classifier on labeled records.
        """
        X_train_list, y_train_list = [], []
        skipped = 0

        for r in records:
            feat_vec, _, err = extract_video_features(r.video_path)
            if feat_vec is None:
                skipped += 1
                continue
            X_train_list.append(feat_vec)
            y_train_list.append(r.label)

        if len(X_train_list) < 4:
            raise ValueError(f"Too few valid training samples extracted ({len(X_train_list)}).")

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        unique_labels = np.unique(y_train)
        if len(unique_labels) < 2:
            raise ValueError(f"Training set must contain both classes; found only {unique_labels}.")

        # Base estimator: standardized logistic regression with L2 regularization
        scaler = StandardScaler()
        base_lr = LogisticRegression(
            C=self.c_regularization,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )

        pipeline = Pipeline([("scaler", scaler), ("classifier", base_lr)])

        # Probability calibration via Platt scaling (sigmoid)
        # Determine internal CV folds
        n_splits = min(3, int(np.min(np.bincount(y_train))))
        if n_splits >= 2:
            calibrated = CalibratedClassifierCV(
                estimator=pipeline,
                method="sigmoid",
                cv=n_splits,
            )
            calibrated.fit(X_train, y_train)
            self.calibrated_model = calibrated
            self.pipeline = pipeline
        else:
            # Fallback to standard uncalibrated logistic regression pipeline if samples per class < 2
            pipeline.fit(X_train, y_train)
            self.pipeline = pipeline
            self.calibrated_model = None

        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": int(len(X_train)),
            "n_skipped": skipped,
            "classes": [int(c) for c in unique_labels],
            "class_counts": {int(c): int(count) for c, count in zip(*np.unique(y_train, return_counts=True))},
            "c_regularization": self.c_regularization,
            "decision_threshold": self.decision_threshold,
            "feature_names": self.feature_names,
            "is_calibrated": self.calibrated_model is not None,
        }
        return self.training_metadata

    def predict_single(
        self,
        video_path: str | Path,
    ) -> dict[str, Any]:
        """
        Run inference on a single video file.
        Returns prediction, probability score, confidence, features, and diagnostic status.
        """
        path = Path(video_path)
        feature_vector, feat_dict, error = extract_video_features(path)

        if feature_vector is None or error is not None:
            return {
                "video": str(path.resolve()),
                "prediction": "UNKNOWN",
                "probability_manipulated": None,
                "probability_genuine": None,
                "confidence": None,
                "threshold": self.decision_threshold,
                "score_interpretation": "Prediction failed due to video reading/extraction error.",
                "status": "ERROR",
                "error_message": error,
                "features_extracted": None,
            }

        if self.calibrated_model is None and self.pipeline is None:
            return {
                "video": str(path.resolve()),
                "prediction": "UNKNOWN",
                "probability_manipulated": None,
                "probability_genuine": None,
                "confidence": None,
                "threshold": self.decision_threshold,
                "score_interpretation": "Model has not been trained or loaded.",
                "status": "ERROR",
                "error_message": "Model not fitted.",
                "features_extracted": feat_dict,
            }

        X = feature_vector.reshape(1, -1)
        model = self.calibrated_model if self.calibrated_model is not None else self.pipeline
        probabilities = model.predict_proba(X)[0]

        # Extract probability for MANIPULATED (class 1)
        classes = list(model.classes_)
        if LABEL_MANIPULATED in classes:
            idx_manip = classes.index(LABEL_MANIPULATED)
            p_manip = float(probabilities[idx_manip])
        else:
            p_manip = float(probabilities[-1])

        p_genuine = float(1.0 - p_manip)
        predicted_label = LABEL_MANIPULATED if p_manip >= self.decision_threshold else LABEL_GENUINE
        prediction_name = LABEL_TO_NAME[predicted_label]
        confidence = float(max(p_manip, p_genuine))

        return {
            "video": str(path.resolve()),
            "prediction": prediction_name,
            "probability_manipulated": round(p_manip, 4),
            "probability_genuine": round(p_genuine, 4),
            "confidence": round(confidence, 4),
            "threshold": self.decision_threshold,
            "score_interpretation": (
                "Calibrated posterior probability P(manipulated | video_features) in [0.0, 1.0]. "
                f"Values >= {self.decision_threshold} predict MANIPULATED."
            ),
            "status": "SUCCESS",
            "error_message": None,
            "features_extracted": feat_dict,
        }

    def save(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> tuple[Path, Path]:
        """Save the trained classifier pipeline and configuration to disk."""
        m_path = Path(model_path)
        c_path = Path(config_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        c_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline,
            "calibrated_model": self.calibrated_model,
            "feature_names": self.feature_names,
            "decision_threshold": self.decision_threshold,
            "c_regularization": self.c_regularization,
            "metadata": self.training_metadata,
        }
        joblib.dump(payload, m_path)

        config_data = {
            "model_type": "LogisticRegression_PlattCalibrated",
            "feature_names": self.feature_names,
            "decision_threshold": self.decision_threshold,
            "c_regularization": self.c_regularization,
            "saved_at": datetime.now().isoformat(),
            "training_metadata": self.training_metadata,
        }
        with c_path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        return m_path, c_path

    @classmethod
    def load(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> BaselineClassifier:
        """Load a saved classifier pipeline from disk."""
        m_path = Path(model_path)
        if not m_path.is_file():
            raise FileNotFoundError(f"Classifier model file not found: {m_path}")

        payload = joblib.load(m_path)
        instance = cls(
            c_regularization=payload.get("c_regularization", 1.0),
            decision_threshold=payload.get("decision_threshold", 0.5),
        )
        instance.pipeline = payload.get("pipeline")
        instance.calibrated_model = payload.get("calibrated_model")
        instance.feature_names = payload.get("feature_names", list(FEATURE_NAMES))
        instance.training_metadata = payload.get("metadata", {})
        return instance


# Clean external interface for pipeline integration
def predict_video(
    video_path: str | Path,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Standalone function interface for deepfake classification.
    Can be called directly by downstream components or an eligibility gate.
    """
    target_model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not target_model_path.is_file():
        return {
            "video": str(Path(video_path).resolve()),
            "prediction": "UNKNOWN",
            "probability_manipulated": None,
            "probability_genuine": None,
            "confidence": None,
            "threshold": threshold,
            "score_interpretation": "Model weights file does not exist.",
            "status": "ERROR",
            "error_message": f"Model file not found at: {target_model_path}",
            "features_extracted": None,
        }

    classifier = BaselineClassifier.load(target_model_path)
    classifier.decision_threshold = threshold
    return classifier.predict_single(video_path)


def main():
    parser = argparse.ArgumentParser(description="SignalGuard Baseline Deepfake Classifier")
    parser.add_argument("--predict", type=str, help="Video path to classify")
    parser.add_argument("--train-manifest", type=str, help="Path to training dataset manifest CSV/JSON")
    parser.add_argument("--val-manifest", type=str, default=None, help="Optional validation manifest")
    parser.add_argument("--output-model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to save trained model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for manipulation probability")

    args = parser.parse_args()

    if args.predict:
        result = predict_video(args.predict, model_path=args.output_model, threshold=args.threshold)
        print(json.dumps(result, indent=2))
        return 0

    if args.train_manifest:
        records = DatasetLoader.load_from_manifest(args.train_manifest)
        print(f"Loaded {len(records)} training records from {args.train_manifest}")

        classifier = BaselineClassifier(decision_threshold=args.threshold)
        meta = classifier.fit(records)
        m_path, c_path = classifier.save(args.output_model)
        print(f"Model successfully saved to: {m_path}")
        print(f"Configuration saved to: {c_path}")
        print("Training metadata:")
        print(json.dumps(meta, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
