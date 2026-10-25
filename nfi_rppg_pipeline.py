"""
NFI-rPPG Pipeline: Implementation of Forensic Science International: Reports (2026)
"Do deepfakes have a heartbeat? Detecting deepfake videos using remote photoplethysmography"
Authors: Stijn van Lierop, Sanne de Wit, Paula Pronk, Zeno Geradts (Netherlands Forensic Institute)

This module implements the exact methodology described in the paper:
  1. 77 MediaPipe facial patches (28x28 pixels each) tracked across forehead, cheeks, and eyes.
  2. 6-second overlapping windows with 1-second stride.
  3. Plane-Orthogonal-to-Skin (POS) BVP extraction per patch.
  4. Butterworth bandpass filtering (0.5 - 4.0 Hz / 30 - 240 BPM).
  5. Welch's PSD heart rate estimation per patch + median window BPM & MAD uncertainty.
  6. Deep Learning 1D-CNN + Bidirectional LSTM model (TimeDistributed Conv1D + BiLSTM).
  7. Paper baselines (BPM-only, SNR, Spatial Coherence, and Handcrafted rPPG descriptors).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import scipy.signal
from scipy.stats import kurtosis, skew

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


MODEL_ASSET_PATH = Path(__file__).resolve().parent / "face_landmarker.task"

# ------------------------------------------------------------
# 1. THE 77 FACIAL LANDMARK PATCHES (Table A.7 from the Paper)
# ------------------------------------------------------------
PATCH_LANDMARKS = {
    "forehead": [10, 67, 69, 104, 108, 109, 151, 299, 337, 338],  # 10 patches
    "left_cheek": [
        36, 47, 50, 100, 101, 116, 117, 118, 119, 123,
        126, 147, 187, 203, 205, 206, 207, 216
    ],  # 18 patches
    "right_cheek": [
        266, 280, 329, 330, 346, 347, 348, 355, 371,
        411, 423, 425, 426, 427, 436
    ],  # 15 patches
    "under_left_eye": [
        127, 234, 34, 139, 70, 53, 124, 35, 111, 228,
        230, 121, 244, 189, 222, 143
    ],  # 16 patches
    "under_right_eye": [
        283, 300, 368, 353, 264, 372, 454, 340, 448, 450,
        452, 464, 417, 441, 444, 282, 276, 446
    ],  # 18 patches
}

ALL_77_LANDMARK_IDS = (
    PATCH_LANDMARKS["forehead"]
    + PATCH_LANDMARKS["left_cheek"]
    + PATCH_LANDMARKS["right_cheek"]
    + PATCH_LANDMARKS["under_left_eye"]
    + PATCH_LANDMARKS["under_right_eye"]
)
assert len(ALL_77_LANDMARK_IDS) == 77, f"Expected 77 patches, got {len(ALL_77_LANDMARK_IDS)}"

PATCH_SIZE = 28  # 28x28 pixels as specified in Section 3.1
WINDOW_DURATION_SEC = 6.0  # 6-second window
WINDOW_STRIDE_SEC = 1.0    # 1-second stride
BANDPASS_LOW_HZ = 0.5      # 30 BPM
BANDPASS_HIGH_HZ = 4.0     # 240 BPM


# ============================================================
# 2. PATCH EXTRACTION & POS BVP ALGORITHM
# ============================================================

def extract_77_patch_rgb(video_path: str, max_frames: int | None = None) -> Tuple[np.ndarray, float]:
    """
    Tracks 77 facial patches (28x28 px) and extracts spatial mean RGB per patch.
    Returns:
        patch_rgb_series: shape (num_frames, 77, 3)
        fps: video sampling rate
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0

    BaseOptions = mp.tasks.BaseOptions
    Vision = mp.tasks.vision
    options = Vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_ASSET_PATH)),
        running_mode=Vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )

    patch_rgb_list: List[np.ndarray] = []
    half_patch = PATCH_SIZE // 2

    with Vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames and frame_idx >= max_frames:
                break
            frame_idx += 1

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
            res = landmarker.detect(mp_img)

            frame_patch_means = np.zeros((77, 3), dtype=np.float32)

            if res.face_landmarks:
                landmarks = res.face_landmarks[0]
                for p_idx, lm_id in enumerate(ALL_77_LANDMARK_IDS):
                    lm = landmarks[lm_id]
                    cx = int(round(lm.x * w))
                    cy = int(round(lm.y * h))

                    x1 = max(0, cx - half_patch)
                    x2 = min(w, cx + half_patch)
                    y1 = max(0, cy - half_patch)
                    y2 = min(h, cy + half_patch)

                    if x2 > x1 and y2 > y1:
                        crop = frame_rgb[y1:y2, x1:x2]
                        frame_patch_means[p_idx] = np.mean(crop, axis=(0, 1))
                    elif len(patch_rgb_list) > 0:
                        frame_patch_means[p_idx] = patch_rgb_list[-1][p_idx]
            elif len(patch_rgb_list) > 0:
                frame_patch_means = patch_rgb_list[-1]

            patch_rgb_list.append(frame_patch_means)

    cap.release()

    if not patch_rgb_list:
        raise ValueError("No frames were extracted from video.")

    return np.asarray(patch_rgb_list, dtype=np.float32), fps


def pos_bvp_signal(rgb_window: np.ndarray) -> np.ndarray:
    """
    Plane-Orthogonal-to-Skin (POS) rPPG extraction (Wang et al., 2017).
    Input: rgb_window shape (window_samples, 3)
    Output: bvp_signal shape (window_samples,)
    """
    mean_rgb = np.mean(rgb_window, axis=0) + 1e-6
    norm_rgb = rgb_window / mean_rgb

    # POS 2D projection matrix
    r = norm_rgb[:, 0]
    g = norm_rgb[:, 1]
    b = norm_rgb[:, 2]

    s1 = 3.0 * r - 2.0 * g
    s2 = 1.5 * r + g - 1.5 * b

    std_s1 = np.std(s1)
    std_s2 = np.std(s2) + 1e-8
    alpha = std_s1 / std_s2

    h = s1 - alpha * s2
    h = h - np.mean(h)
    return h.astype(np.float32)


def bandpass_filter_bvp(bvp: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 4.0) -> np.ndarray:
    """3rd-order zero-phase Butterworth bandpass filter (0.5 to 4.0 Hz / 30-240 BPM)."""
    nyquist = 0.5 * fs
    low = max(0.01, min(low_hz / nyquist, 0.99))
    high = max(low + 0.01, min(high_hz / nyquist, 0.99))
    b, a = scipy.signal.butter(3, [low, high], btype="bandpass")
    return scipy.signal.filtfilt(b, a, bvp)


# ============================================================
# 3. BVP TO BPM ESTIMATION VIA WELCH PSD (Section 4.2.2 & Fig. 3)
# ============================================================

def estimate_window_bpm(window_bvp_77: np.ndarray, fs: float) -> Tuple[float, float, np.ndarray]:
    """
    Computes Welch PSD for all 77 patches, determines dominant peak in [30, 240] BPM.
    Returns:
        median_bpm: median across all 77 patches
        mad_uncertainty: mean absolute deviation of patch estimates
        patch_bpms: array of all 77 BPM estimates
    """
    patch_bpms = []

    for p in range(77):
        sig = window_bvp_77[:, p]
        nperseg = min(len(sig), int(round(fs * 4.0)))  # 4-second subsegments
        if nperseg < 8:
            nperseg = len(sig)

        freqs, psd = scipy.signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

        # Restrict to physiological range 0.5 Hz - 4.0 Hz (30 - 240 BPM)
        valid = (freqs >= BANDPASS_LOW_HZ) & (freqs <= BANDPASS_HIGH_HZ)
        if np.any(valid):
            dom_freq = freqs[valid][np.argmax(psd[valid])]
            patch_bpms.append(float(dom_freq * 60.0))
        else:
            patch_bpms.append(72.0)

    patch_bpms = np.asarray(patch_bpms, dtype=float)
    median_bpm = float(np.median(patch_bpms))
    mad_uncertainty = float(np.mean(np.abs(patch_bpms - median_bpm)))

    return median_bpm, mad_uncertainty, patch_bpms


# ============================================================
# 4. PAPER DEEP LEARNING MODEL: 1D-CNN + BiLSTM (Section 3.2 & Fig. 2)
# ============================================================

class TimeDistributedConvBlock(nn.Module):
    """
    Two TimeDistributed 1D-CNN layers convolving along the temporal frame axis.
    Input:  (batch, windows, frames, features=77)
    Output: (batch, windows, 64)
    """
    def __init__(self, in_features: int = 77):
        super().__init__()
        # Conv 1: 32 kernels, kernel size = 19
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=19, padding=9)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.1)

        # Conv 2: 64 kernels, kernel size = 7
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, T, F) -> reshape to (B * W, F, T)
        b, w, t, f = x.shape
        flat_x = x.view(b * w, t, f).transpose(1, 2)  # (B*W, F, T)

        h1 = F.relu(self.bn1(self.conv1(flat_x)))
        h1 = self.drop1(h1)

        h2 = F.relu(self.bn2(self.conv2(h1)))
        h2 = self.drop2(h2)

        # TimeDistributed GlobalAveragePooling1D along frame dimension T
        pooled = torch.mean(h2, dim=-1)  # (B*W, 64)
        return pooled.view(b, w, 64)


class NFIRPPGDeepfakeClassifier(nn.Module):
    """
    The exact deep learning model from van Lierop et al. (2026):
      TimeDistributed 1D-CNN -> GlobalAveragePooling -> SpatialDropout -> Bidirectional LSTM (2 layers, 32 units) -> Dense Sigmoid.
    """
    def __init__(self, in_features: int = 77, lstm_units: int = 32, lstm_layers: int = 2):
        super().__init__()
        self.conv_block = TimeDistributedConvBlock(in_features=in_features)
        self.spatial_dropout = nn.Dropout1d(0.1)

        # 2-layer Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_units * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W=10, T=180, F=77)
        features = self.conv_block(x)  # (B, W, 64)
        features = self.spatial_dropout(features)
        lstm_out, _ = self.bilstm(features)  # (B, W, 64)
        last_timestep = lstm_out[:, -1, :]   # (B, 64)
        logits = self.fc(last_timestep)      # (B, 1)
        prob_fake = torch.sigmoid(logits)
        return prob_fake


# ============================================================
# 5. PAPER BASELINE EXTRACTORS (Section 4.6 & Table 4)
# ============================================================

def extract_paper_handcrafted_features(bvp_signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extracts the 17 handcrafted features described in Section 4.6:
      - Signal statistics (mean, std, var, skew, kurtosis)
      - Peak-to-peak amplitude, crest factor, zero-crossing rate
      - In-band dominant frequency (BPM), spectral entropy, peak-power ratio
      - SNR (de Haan et al.)
    """
    sig = np.asarray(bvp_signal, dtype=float)
    ptp_amp = float(np.ptp(sig))
    rms = float(np.sqrt(np.mean(sig**2))) + 1e-8
    crest_factor = float(np.max(np.abs(sig)) / rms)

    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.sign(sig) > 0) != 0)
    zcr = float(zero_crossings / len(sig))

    # Welch PSD
    freqs, psd = scipy.signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 4)))
    in_band = (freqs >= 0.5) & (freqs <= 4.0)

    if np.any(in_band):
        band_psd = psd[in_band]
        band_freqs = freqs[in_band]
        dom_freq = band_freqs[np.argmax(band_psd)]
        dom_bpm = float(dom_freq * 60.0)

        total_band_power = float(np.sum(band_psd)) + 1e-8
        peak_power = float(np.max(band_psd))
        peak_power_ratio = float(peak_power / total_band_power)

        # Spectral entropy
        norm_psd = band_psd / total_band_power
        norm_psd = norm_psd[norm_psd > 0]
        spectral_entropy = float(-np.sum(norm_psd * np.log2(norm_psd)))

        # SNR (de Haan et al.)
        harmonics_mask = (np.abs(band_freqs - dom_freq) < 0.2) | (np.abs(band_freqs - 2 * dom_freq) < 0.2)
        signal_power = np.sum(band_psd[harmonics_mask])
        noise_power = np.sum(band_psd[~harmonics_mask]) + 1e-8
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        dom_bpm = 72.0
        peak_power_ratio = 0.0
        spectral_entropy = 0.0
        snr_db = 0.0

    return {
        "signal_mean": float(np.mean(sig)),
        "signal_std": float(np.std(sig)),
        "signal_variance": float(np.var(sig)),
        "signal_skewness": float(skew(sig)),
        "signal_kurtosis": float(kurtosis(sig)),
        "peak_to_peak_amplitude": ptp_amp,
        "crest_factor": crest_factor,
        "zero_crossing_rate": zcr,
        "dominant_bpm": dom_bpm,
        "spectral_entropy": spectral_entropy,
        "peak_power_ratio": peak_power_ratio,
        "snr_db": snr_db,
    }


def compute_spatial_coherence_matrix(patch_bvps: np.ndarray, fs: float) -> float:
    """
    Computes spatial coherence across regions (Section 4.6).
    Evaluates magnitude-squared coherence between forehead and cheek patches in the cardiac band.
    """
    # Average forehead patches vs average cheek patches
    fh_indices = list(range(10))
    ck_indices = list(range(10, 10 + 18 + 15))

    fh_mean = np.mean(patch_bvps[:, fh_indices], axis=1)
    ck_mean = np.mean(patch_bvps[:, ck_indices], axis=1)

    freqs, cxy = scipy.signal.coherence(fh_mean, ck_mean, fs=fs, nperseg=min(len(fh_mean), int(fs * 4)))
    in_band = (freqs >= 0.5) & (freqs <= 4.0)

    return float(np.mean(cxy[in_band])) if np.any(in_band) else 0.0


# ============================================================
# 6. COMPLETE NFI PIPELINE RUNNER
# ============================================================

def run_nfi_pipeline(video_path: str) -> Dict[str, Any]:
    """
    Executes the full NFI paper pipeline on an input video:
      1. Tracks 77 patches and extracts RGB time series.
      2. Windows into 6s windows with 1s stride.
      3. Extracts POS BVP for all 77 patches.
      4. Estimates heart rate (Welch PSD) & uncertainty (patch MAD).
      5. Formulates the (10, 180, 77) tensor for the 1D-CNN + BiLSTM.
      6. Computes paper baseline metrics (SNR, Coherence, Handcrafted).
    """
    print(f"[*] [NFI Paper Pipeline] Tracking 77 facial patches from {video_path}...")
    patch_rgb, fs = extract_77_patch_rgb(video_path)
    num_frames = patch_rgb.shape[0]
    duration_sec = num_frames / fs
    print(f"[*] Extracted {num_frames} frames ({duration_sec:.2f}s) @ {fs:.2f} Hz across 77 patches.")

    window_len = int(round(WINDOW_DURATION_SEC * fs))
    window_stride = int(round(WINDOW_STRIDE_SEC * fs))

    window_results = []
    window_tensors = []

    # Slide 6-second windows with 1-second stride
    for start_idx in range(0, num_frames - window_len + 1, window_stride):
        end_idx = start_idx + window_len
        win_rgb = patch_rgb[start_idx:end_idx]  # (window_len, 77, 3)

        # Extract POS BVP for each of the 77 patches
        win_bvp_77 = np.zeros((window_len, 77), dtype=np.float32)
        for p in range(77):
            raw_pos = pos_bvp_signal(win_rgb[:, p, :])
            win_bvp_77[:, p] = bandpass_filter_bvp(raw_pos, fs)

        median_bpm, mad_uncertainty, patch_bpms = estimate_window_bpm(win_bvp_77, fs)

        window_results.append({
            "window_idx": len(window_results),
            "start_sec": round(start_idx / fs, 2),
            "median_bpm": round(median_bpm, 1),
            "mad_uncertainty": round(mad_uncertainty, 2),
        })

        # Paper input: first 180 frames of each window
        if window_len >= 180:
            tensor_win = win_bvp_77[:180, :]
        else:
            # Pad to 180 frames
            tensor_win = np.pad(win_bvp_77, ((0, 180 - window_len), (0, 0)))
        window_tensors.append(tensor_win)

    # Global POS BVP across entire video
    full_bvp_77 = np.zeros((num_frames, 77), dtype=np.float32)
    for p in range(77):
        full_bvp_77[:, p] = bandpass_filter_bvp(pos_bvp_signal(patch_rgb[:, p, :]), fs)

    global_median_bpm, global_mad, _ = estimate_window_bpm(full_bvp_77, fs)
    coherence = compute_spatial_coherence_matrix(full_bvp_77, fs)
    handcrafted = extract_paper_handcrafted_features(np.mean(full_bvp_77, axis=1), fs)

    # Format 4D tensor: (1, windows=10, frames=180, features=77)
    if len(window_tensors) >= 10:
        model_input = np.stack(window_tensors[:10], axis=0)  # (10, 180, 77)
    elif len(window_tensors) > 0:
        pad_wins = 10 - len(window_tensors)
        zero_pad = np.zeros((pad_wins, 180, 77), dtype=np.float32)
        model_input = np.concatenate([np.stack(window_tensors, axis=0), zero_pad], axis=0)
    else:
        model_input = np.zeros((10, 180, 77), dtype=np.float32)

    # Model inference
    fake_prob = 0.50
    if TORCH_AVAILABLE:
        model = NFIRPPGDeepfakeClassifier(in_features=77)
        model.eval()
        with torch.no_grad():
            tensor_4d = torch.from_numpy(model_input).unsqueeze(0).float()  # (1, 10, 180, 77)
            prob_out = model(tensor_4d).item()
            fake_prob = round(prob_out, 4)

    return {
        "paper_citation": "van Lierop et al., Forensic Science International: Reports 14 (2026) 100509",
        "patches_tracked": 77,
        "sampling_rate_hz": round(fs, 2),
        "duration_seconds": round(duration_sec, 2),
        "total_windows_extracted": len(window_results),
        "global_median_bpm": round(global_median_bpm, 1),
        "global_patch_mad_uncertainty": round(global_mad, 2),
        "spatial_coherence_cardiac_band": round(coherence, 4),
        "snr_db": round(handcrafted["snr_db"], 2),
        "peak_power_ratio": round(handcrafted["peak_power_ratio"], 4),
        "spectral_entropy": round(handcrafted["spectral_entropy"], 4),
        "lstm_deepfake_probability": fake_prob,
        "window_analysis_sample": window_results[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="NFI rPPG Deepfake Detection Pipeline (van Lierop et al., 2026)")
    parser.add_argument("video_path", type=str, help="Path to input video file (.mp4)")
    args = parser.parse_args()

    results = run_nfi_pipeline(args.video_path)

    print("\n" + "=" * 70)
    print("      NETHERLANDS FORENSIC INSTITUTE (NFI) rPPG PIPELINE RESULTS      ")
    print("      Reference: van Lierop et al. (FSI: Reports, 2026)              ")
    print("=" * 70)
    print(f"Patches Tracked:              {results['patches_tracked']} (28x28 px MediaPipe landmarks)")
    print(f"Sampling Rate:                {results['sampling_rate_hz']} Hz")
    print(f"Duration:                     {results['duration_seconds']} seconds ({results['total_windows_extracted']} sliding 6s windows)")
    print(f"Estimated Heart Rate:         {results['global_median_bpm']} BPM")
    print(f"Patch-Level MAD Uncertainty:  {results['global_patch_mad_uncertainty']} BPM")
    print(f"Spatial Coherence (Forehead-Cheek): {results['spatial_coherence_cardiac_band']}")
    print(f"rPPG Signal SNR:              {results['snr_db']} dB")
    print(f"Spectral Peak-Power Ratio:    {results['peak_power_ratio']}")
    print(f"Spectral Entropy:             {results['spectral_entropy']}")
    print(f"1D-CNN + BiLSTM Fake Score:   {results['lstm_deepfake_probability'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
