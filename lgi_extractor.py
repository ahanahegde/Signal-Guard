"""
SignalGuard: LGI-based rPPG extraction.

This script replaces the POS calculation with a Local Group Invariance (LGI)
style projection while preserving the legacy *_POS and *_POS_filtered CSV
columns expected by the current eligibility gate and stress runner.

Important:
- The legacy *_POS columns contain LGI output in this script. They are retained
  for compatibility only; they are not computed with POS.
- This is an experimental implementation. Validate it against a synchronized
  contact PPG/ECG reference before treating BPM estimates as accurate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from signalguard_utils import (
    infer_sampling_rate,
    detect_rgb_artifacts,
    filter_pos_signal,
    analyse_signal,
)

RGB_PATH = Path("rgb_signals.csv")
OUTPUT_CSV = Path("pos_signals.csv")  # Keep the filename expected by existing stages.
OUTPUT_PLOT = Path("pos_signals.png")
REGIONS = {
    "forehead": ["forehead_R", "forehead_G", "forehead_B"],
    "cheeks": ["cheeks_R", "cheeks_G", "cheeks_B"],
}
ARTIFACT_THRESHOLD = 0.10


def interpolate_invalid_rows(rgb: np.ndarray) -> np.ndarray:
    """Fill invalid RGB rows by per-channel linear interpolation for processing."""
    data = np.asarray(rgb, dtype=float).copy()
    n = len(data)
    x = np.arange(n)
    for channel in range(data.shape[1]):
        values = data[:, channel]
        good = np.isfinite(values) & (values > 1e-6)
        if good.sum() < 2:
            raise ValueError("Not enough valid RGB samples to interpolate.")
        values[~good] = np.interp(x[~good], x[good], values[good])
        data[:, channel] = values
    return data


def lgi_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Extract an LGI-style pulse candidate from one RGB region.

    Steps:
      1. Normalize each colour channel by its temporal mean.
      2. Remove the temporal mean.
      3. Estimate the dominant colour direction with SVD.
      4. Project RGB traces onto the plane orthogonal to that direction.
      5. Select the strongest projected component using PCA.

    The result is a pulse candidate, not a verified physiological measurement.
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("RGB input must have shape (frames, 3).")
    if len(rgb) < 10:
        raise ValueError("Not enough RGB frames for LGI extraction.")

    clean = interpolate_invalid_rows(rgb)
    channel_means = np.mean(clean, axis=0)
    channel_means = np.where(np.abs(channel_means) < 1e-8, 1.0, channel_means)

    # Normalized colour traces, shaped 3 x time.
    X = (clean / channel_means).T
    X = X - np.mean(X, axis=1, keepdims=True)

    # Dominant direction captures common colour variation (often illumination).
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    dominant = U[:, 0]
    projection = np.eye(3) - np.outer(dominant, dominant)
    projected = projection @ X

    # PCA of projected traces; choose the strongest remaining component.
    covariance = projected @ projected.T / max(projected.shape[1] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    component = eigvecs[:, int(np.argmax(eigvals))]
    signal = component @ projected

    signal = signal - np.mean(signal)
    scale = np.std(signal)
    if not np.isfinite(scale) or scale < 1e-10:
        return np.full(len(signal), np.nan)
    return signal / scale


if not RGB_PATH.exists():
    raise SystemExit(
        "ERROR: rgb_signals.csv not found. Run signal_extractor.py first."
    )

df = pd.read_csv(RGB_PATH)
if len(df) == 0:
    raise SystemExit("ERROR: RGB CSV contains no frames.")
if "time_seconds" not in df.columns:
    raise SystemExit(
        "ERROR: time_seconds is missing. Run the current signal_extractor.py first."
    )

try:
    fs = infer_sampling_rate(df)
except ValueError as error:
    raise SystemExit(f"ERROR: {error}")

times = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(dtype=float)
if len(times) > 1:
    duration = float(times[-1] - times[0])
else:
    duration = 0.0

print("=" * 56)
print("SIGNALGUARD - LGI rPPG EXTRACTION (experimental)")
print("=" * 56)
print(f"Frames processed: {len(df)}")
print(f"Timestamp-derived sampling rate: {fs:.4f} Hz")
print(f"Duration: {duration:.2f} seconds")
print("Compatibility note: *_POS columns below contain LGI output.")

for region, columns in REGIONS.items():
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: Missing {region} RGB columns: {missing}")

    rgb = df[columns].to_numpy(dtype=float)
    original_valid = (
        np.all(np.isfinite(rgb), axis=1) & np.all(rgb > 1e-6, axis=1)
    )

    artifact_mask, relative_change = detect_rgb_artifacts(
        rgb, threshold=ARTIFACT_THRESHOLD
    )
    cleaned_rgb = rgb.copy()
    cleaned_rgb[artifact_mask] = np.nan

    try:
        lgi = lgi_from_rgb(cleaned_rgb)
    except ValueError as error:
        raise SystemExit(f"ERROR: LGI failed for {region}: {error}")

    # Reuse the project's existing filter and analysis functions so the
    # downstream gate sees the same filtering/metric conventions as before.
    filtered = filter_pos_signal(lgi, fs)
    metrics = analyse_signal(filtered, fs)

    clean_valid = np.all(np.isfinite(cleaned_rgb), axis=1)
    filtered_valid = np.isfinite(filtered)

    df[f"{region}_RGB_artifact"] = artifact_mask
    df[f"{region}_RGB_relative_change"] = relative_change

    # Native LGI names.
    df[f"{region}_LGI"] = lgi
    df[f"{region}_LGI_filtered"] = filtered

    # Legacy names expected by the current eligibility gate.
    df[f"{region}_POS"] = lgi
    df[f"{region}_POS_filtered"] = filtered

    print(f"\n{region.upper()}")
    print(f"Original RGB coverage: {original_valid.mean() * 100:.1f}%")
    print(f"RGB coverage after artifact mask: {clean_valid.mean() * 100:.1f}%")
    print(f"Flagged artifact samples: {int(np.sum(artifact_mask))}")
    print(f"Filtered signal coverage: {filtered_valid.mean() * 100:.1f}%")
    if metrics["bpm"] is None:
        print("Candidate BPM: unavailable")
    else:
        print(f"Candidate BPM: {metrics['bpm']:.1f}")
        print(f"Peak frequency: {metrics['frequency_hz']:.4f} Hz")
        print(f"Spectral peak ratio: {metrics['peak_ratio']:.2f}")
        if metrics["multi_peak_ratio"] is None:
            print("Competing peak: no distinct competitor found")
        else:
            print(
                "Dominant/competing peak ratio: "
                f"{metrics['multi_peak_ratio']:.2f}"
            )

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV saved: {OUTPUT_CSV}")

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
for row, region in enumerate(["forehead", "cheeks"]):
    axes[row, 0].plot(times, df[f"{region}_LGI"], linewidth=1)
    axes[row, 0].set_title(f"{region.title()} Raw LGI")
    axes[row, 1].plot(times, df[f"{region}_LGI_filtered"], linewidth=1)
    axes[row, 1].set_title(f"{region.title()} Filtered LGI")
    axes[row, 0].set_ylabel("Normalized signal")
    axes[row, 1].set_ylabel("Normalized signal")
    axes[row, 0].grid(True, alpha=0.3)
    axes[row, 1].grid(True, alpha=0.3)

axes[1, 0].set_xlabel("Time (seconds)")
axes[1, 1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"Plot saved: {OUTPUT_PLOT}")

# Runner uses a non-interactive Matplotlib backend; show() is harmless there.
plt.show()
