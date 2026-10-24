"""
SignalGuard POS extractor — windowed POS with diagnostic checks.

Maintains full compatibility with existing SignalGuard CSV schema:
  forehead_POS, forehead_POS_filtered, forehead_POS_windowed
  cheeks_POS, cheeks_POS_filtered, cheeks_POS_windowed
  forehead_RGB_artifact, forehead_RGB_relative_change
  cheeks_RGB_artifact, cheeks_RGB_relative_change
and writes pos_signals.csv / pos_signals.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from signalguard_utils import (
    infer_sampling_rate,
    detect_rgb_artifacts,
    pos_from_rgb,
    filter_pos_signal,
    spectral_metrics,
    analyse_signal,
    LOW_HZ,
    HIGH_HZ,
    POS_WINDOW_SECONDS,
)

RGB_PATH = Path("rgb_signals.csv")
OUTPUT_CSV = Path("pos_signals.csv")
OUTPUT_PLOT = Path("pos_signals.png")

REGIONS = {
    "forehead": ["forehead_R", "forehead_G", "forehead_B"],
    "cheeks": ["cheeks_R", "cheeks_G", "cheeks_B"],
}

ARTIFACT_THRESHOLD = 0.10


def main():
    if not RGB_PATH.exists():
        raise SystemExit(
            "ERROR: rgb_signals.csv not found. Run signal_extractor.py first."
        )

    df = pd.read_csv(RGB_PATH)
    if df.empty:
        raise SystemExit("ERROR: RGB CSV contains no frames.")
    if "time_seconds" not in df.columns:
        raise SystemExit(
            "ERROR: time_seconds is missing. Run signal_extractor.py first."
        )

    try:
        fs = float(infer_sampling_rate(df))
    except (ValueError, TypeError) as error:
        raise SystemExit(f"ERROR: Could not infer sampling rate: {error}")

    if not np.isfinite(fs) or fs <= 2 * LOW_HZ:
        raise SystemExit(f"ERROR: Invalid sampling rate: {fs}")

    times = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(times)):
        raise SystemExit("ERROR: time_seconds contains invalid values.")
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    # Extract tracking motion signals if present in df
    motion_cols = [c for c in ["face_center_y_norm", "face_center_x_norm", "face_width_norm"] if c in df.columns]
    motion_signals = df[motion_cols].to_numpy(dtype=float) if motion_cols else None

    print("=" * 58)
    print("SIGNALGUARD - WINDOWED POS EXTRACTION")
    print("=" * 58)
    print(f"Frames processed: {len(df)}")
    print(f"Timestamp-derived sampling rate: {fs:.4f} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"POS window: {POS_WINDOW_SECONDS:.1f} s")
    print(f"Band-pass: {LOW_HZ:.2f}-{HIGH_HZ:.2f} Hz")

    plots = {}
    results = {}
    for region, columns in REGIONS.items():
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise SystemExit(f"ERROR: Missing {region} RGB columns: {missing}")

        rgb = df[columns].to_numpy(dtype=float)
        original_valid = (
            np.all(np.isfinite(rgb), axis=1) & np.all(rgb > 1e-6, axis=1)
        )

        try:
            artifact_mask, relative_change = detect_rgb_artifacts(
                rgb, threshold=ARTIFACT_THRESHOLD
            )
            artifact_mask = np.asarray(artifact_mask, dtype=bool)

            clean_rgb = rgb.copy()
            clean_rgb[artifact_mask, :] = np.nan
            post_mask_valid = np.all(np.isfinite(clean_rgb), axis=1)

            # Unified standard POS extraction & filtering from signalguard_utils
            raw_pos = pos_from_rgb(clean_rgb, fs, window_seconds=POS_WINDOW_SECONDS)
            filtered_pos = filter_pos_signal(raw_pos, fs, low=LOW_HZ, high=HIGH_HZ)
            metrics = analyse_signal(filtered_pos, fs, motion_signals=motion_signals)
        except Exception as error:
            raise SystemExit(f"ERROR: extraction failed for {region}: {error}")

        df[f"{region}_RGB_artifact"] = artifact_mask
        df[f"{region}_RGB_relative_change"] = relative_change

        df[f"{region}_POS"] = raw_pos
        df[f"{region}_POS_filtered"] = filtered_pos
        df[f"{region}_POS_windowed"] = raw_pos

        results[region] = metrics
        plots[region] = (raw_pos, filtered_pos)

        print(f"\n{region.upper()}")
        print(f"Original RGB coverage: {original_valid.mean() * 100:.1f}%")
        print(f"RGB coverage after artifact mask: {post_mask_valid.mean() * 100:.1f}%")
        print(f"Flagged artifact samples: {int(artifact_mask.sum())}")

        if metrics["bpm"] is None:
            print("Candidate BPM: unavailable")
        else:
            print(f"Candidate BPM: {metrics['bpm']:.1f}")
            print(f"Peak frequency: {metrics['frequency_hz']:.4f} Hz")
            print(f"Spectral peak ratio: {metrics['peak_ratio']:.2f}")
            if metrics["multi_peak_ratio"] is None:
                print("Dominant/competing spectral peak ratio: no distinct competitor")
            else:
                print(f"Dominant/competing spectral peak ratio: {metrics['multi_peak_ratio']:.2f}")

            if metrics["half_rate_power_ratio"] is not None:
                print(f"Power ratio at half the dominant frequency: {metrics['half_rate_power_ratio']:.2f}")

            if metrics.get("candidate_peaks"):
                cand_strs = [
                    f"{c['bpm']:.1f} BPM ({c['frequency_hz']:.3f} Hz" + (", motion" if c.get("is_motion") else "") + ")"
                    for c in metrics["candidate_peaks"]
                ]
                print("Candidate spectral peaks: " + "; ".join(cand_strs))

            if metrics.get("motion_frequencies_hz"):
                print(f"Detected tracking motion peaks: {', '.join(f'{f:.3f} Hz ({f*60:.1f} BPM)' for f in metrics['motion_frequencies_hz'])}")

    # Cross-region diagnostic check
    a, b = results["forehead"]["bpm"], results["cheeks"]["bpm"]
    print("\nCROSS-REGION CHECK")
    if a is not None and b is not None:
        difference = abs(a - b)
        print(f"Forehead-cheek candidate difference: {difference:.1f} BPM")
        if difference > 13.0:
            print("WARNING: regions disagree; inspect candidate peaks and motion diagnostics.")
        else:
            print("Regional estimates agree within threshold, but this is not ground truth.")
    print("A plausible waveform or close regional agreement does not prove authenticity.")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV saved: {OUTPUT_CSV}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for row, region in enumerate(["forehead", "cheeks"]):
        raw_pos, filtered_pos = plots[region]
        axes[row, 0].plot(times, raw_pos, linewidth=1)
        axes[row, 0].set_title(f"{region.title()} Raw Windowed POS")
        axes[row, 1].plot(times, filtered_pos, linewidth=1)
        axes[row, 1].set_title(f"{region.title()} Filtered POS")
        axes[row, 0].set_ylabel("Normalized signal")
        axes[row, 1].set_ylabel("Filtered signal")
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Time (seconds)")
    axes[1, 1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
