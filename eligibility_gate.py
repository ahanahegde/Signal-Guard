
import numpy as np
import pandas as pd

from pathlib import Path

from signalguard_utils import (
    infer_sampling_rate,
    analyse_signal,
    peak_is_clear,
)

# ============================================================
# SIGNALGUARD - FINAL PROTOTYPE ELIGIBILITY GATE
# ============================================================

POS_FILE = Path("pos_signals.csv")
RGB_FILE = Path("rgb_signals.csv")
WINDOW_OUTPUT = Path("window_analysis.csv")

# Provisional thresholds; these are not scientifically validated.
MIN_COVERAGE = 80.0
MIN_CONTINUOUS_SECONDS = 8.0
MAX_REGION_HR_DIFFERENCE = 13.0

WINDOW_SECONDS = 10.0
WINDOW_STEP_SECONDS = 2.0

MIN_USABLE_WINDOWS = 3
MIN_WINDOW_COVERAGE = 80.0

# Required proportion of usable windows meeting each criterion.
MIN_CONSISTENT_WINDOW_FRACTION = 0.60
MIN_CLEAR_PEAK_WINDOW_FRACTION = 0.60

# Allowed median absolute deviation of window BPM estimates.
MAX_WINDOW_BPM_MAD = 12.0


def median_absolute_deviation(values):
    """Robust spread of window estimates around their median."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def display_metrics(name, metrics):
    print(f"\n{name}")
    print(f"  Coverage: {metrics['coverage']:.1f}%")
    print(
        f"  Longest continuous segment: "
        f"{metrics['longest_seconds']:.2f} seconds"
    )

    if metrics["bpm"] is None:
        print("  Candidate BPM: unavailable")
        print("  Peak ratio: unavailable")
        return

    print(f"  Candidate BPM: {metrics['bpm']:.1f}")
    print(f"  Peak frequency: {metrics['frequency_hz']:.4f} Hz")
    print(f"  Peak/mean power ratio: {metrics['peak_ratio']:.2f}")

    if metrics["multi_peak_ratio"] is None:
        print("  Competing-peak ratio: no distinct competitor")
    else:
        print(
            "  Dominant/competing-peak ratio: "
            f"{metrics['multi_peak_ratio']:.2f}"
        )

    if metrics.get("candidate_peaks"):
        cand_str = "; ".join(
            f"{c['bpm']:.1f} BPM ({c['frequency_hz']:.3f} Hz" + (", motion" if c.get("is_motion") else "") + ")"
            for c in metrics["candidate_peaks"][:4]
        )
        print(f"  Candidate peaks: {cand_str}")
    if metrics.get("motion_frequencies_hz"):
        print(f"  Tracking motion detected: {', '.join(f'{f:.3f} Hz ({f*60:.1f} BPM)' for f in metrics['motion_frequencies_hz'])}")


def safe_difference(a, b):
    """Return the absolute difference when both estimates exist."""
    if a is None or b is None:
        return None

    return abs(float(a) - float(b))


# ------------------------------------------------------------
# LOAD AND VALIDATE INPUTS
# ------------------------------------------------------------

for path in (POS_FILE, RGB_FILE):
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} is missing. Run signal_extractor.py, "
            "then pos_extractor.py, before the eligibility gate."
        )

pos = pd.read_csv(POS_FILE)
rgb = pd.read_csv(RGB_FILE)

required_pos = [
    "time_seconds",
    "forehead_POS_filtered",
    "cheeks_POS_filtered",
]

required_rgb = [
    "time_seconds",
    "forehead_R", "forehead_G", "forehead_B",
    "cheeks_R", "cheeks_G", "cheeks_B",
]

for column in required_pos:
    if column not in pos.columns:
        raise SystemExit(f"ERROR: Missing POS column: {column}")

for column in required_rgb:
    if column not in rgb.columns:
        raise SystemExit(f"ERROR: Missing RGB column: {column}")

if len(pos) != len(rgb):
    raise SystemExit(
        "ERROR: POS and RGB row counts differ. "
        "Re-run signal_extractor.py and pos_extractor.py."
    )

pos_times = pd.to_numeric(
    pos["time_seconds"], errors="coerce"
).to_numpy(dtype=float)

rgb_times = pd.to_numeric(
    rgb["time_seconds"], errors="coerce"
).to_numpy(dtype=float)

if not (
    np.all(np.isfinite(pos_times))
    and np.all(np.isfinite(rgb_times))
):
    raise SystemExit("ERROR: Invalid or missing timestamps.")

if not np.allclose(pos_times, rgb_times, atol=1e-4, rtol=0):
    raise SystemExit(
        "ERROR: POS and RGB timestamps differ. "
        "Re-run both extraction scripts on the same video."
    )

try:
    fs = infer_sampling_rate(pos)
except ValueError as error:
    raise SystemExit(f"ERROR: {error}")

if len(pos) < 1:
    raise SystemExit("ERROR: No samples found.")

# ------------------------------------------------------------
# LOAD SIGNALS AND CALCULATE COVERAGE
# ------------------------------------------------------------

forehead_signal = pd.to_numeric(
    pos["forehead_POS_filtered"], errors="coerce"
).to_numpy(dtype=float)

cheeks_signal = pd.to_numeric(
    pos["cheeks_POS_filtered"], errors="coerce"
).to_numpy(dtype=float)

forehead_rgb_valid = rgb[
    ["forehead_R", "forehead_G", "forehead_B"]
].notna().all(axis=1).to_numpy()

cheeks_rgb_valid = rgb[
    ["cheeks_R", "cheeks_G", "cheeks_B"]
].notna().all(axis=1).to_numpy()

rgb_coverage = float(
    np.mean(forehead_rgb_valid & cheeks_rgb_valid) * 100
)

motion_cols = [c for c in ["face_center_y_norm", "face_center_x_norm", "face_width_norm"] if c in rgb.columns]
motion_signals = rgb[motion_cols].to_numpy(dtype=float) if motion_cols else None

forehead = analyse_signal(forehead_signal, fs, motion_signals=motion_signals)
cheeks = analyse_signal(cheeks_signal, fs, motion_signals=motion_signals)

# ------------------------------------------------------------
# ANALYSE THE LONGEST SHARED VALID INTERVAL
# ------------------------------------------------------------

shared_valid = (
    np.isfinite(forehead_signal) &
    np.isfinite(cheeks_signal)
)

shared_forehead = np.where(
    shared_valid, forehead_signal, np.nan
)
shared_cheeks = np.where(
    shared_valid, cheeks_signal, np.nan
)

shared_f_metrics = analyse_signal(shared_forehead, fs, motion_signals=motion_signals)
shared_c_metrics = analyse_signal(shared_cheeks, fs, motion_signals=motion_signals)

shared_difference = safe_difference(
    shared_f_metrics["bpm"],
    shared_c_metrics["bpm"],
)

# ------------------------------------------------------------
# SLIDING-WINDOW ANALYSIS
# ------------------------------------------------------------

window_size = int(round(WINDOW_SECONDS * fs))
step_size = max(1, int(round(WINDOW_STEP_SECONDS * fs)))

n = len(pos)
starts = []

if window_size > 0 and n >= window_size:
    starts = list(range(0, n - window_size + 1, step_size))

    # Include the final possible window instead of leaving a tail out.
    final_start = n - window_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

window_results = []

for start in starts:
    end = start + window_size

    f_window = forehead_signal[start:end].copy()
    c_window = cheeks_signal[start:end].copy()

    # Both regions must be valid at the same time.
    common = np.isfinite(f_window) & np.isfinite(c_window)

    f_window[~common] = np.nan
    c_window[~common] = np.nan

    common_coverage = float(np.mean(common) * 100)

    win_motion = motion_signals[start:end] if motion_signals is not None else None
    f_metrics = analyse_signal(f_window, fs, motion_signals=win_motion, min_hr_seconds=MIN_CONTINUOUS_SECONDS)
    c_metrics = analyse_signal(c_window, fs, motion_signals=win_motion, min_hr_seconds=MIN_CONTINUOUS_SECONDS)

    both_estimates_available = (
        f_metrics["bpm"] is not None
        and c_metrics["bpm"] is not None
    )

    difference = safe_difference(
        f_metrics["bpm"], c_metrics["bpm"]
    )

    usable = (
        common_coverage >= MIN_WINDOW_COVERAGE
        and f_metrics["longest_seconds"] >= MIN_CONTINUOUS_SECONDS
        and c_metrics["longest_seconds"] >= MIN_CONTINUOUS_SECONDS
        and both_estimates_available
    )

    both_peaks_clear = (
        peak_is_clear(f_metrics)
        and peak_is_clear(c_metrics)
    )

    consistent = (
        usable
        and difference is not None
        and difference <= MAX_REGION_HR_DIFFERENCE
    )

    window_results.append({
        "start_seconds": float(pos_times[start]),
        "end_seconds": float(pos_times[end - 1]),
        "common_coverage_pct": common_coverage,
        "forehead_bpm": f_metrics["bpm"],
        "cheek_bpm": c_metrics["bpm"],
        "bpm_difference": difference,
        "forehead_peak_ratio": f_metrics["peak_ratio"],
        "cheek_peak_ratio": c_metrics["peak_ratio"],
        "forehead_multi_peak_ratio": f_metrics["multi_peak_ratio"],
        "cheek_multi_peak_ratio": c_metrics["multi_peak_ratio"],
        "usable": usable,
        "both_peaks_clear": both_peaks_clear,
        "regions_consistent": consistent,
    })

windows_df = pd.DataFrame(window_results)

usable_windows = (
    [w for w in window_results if w["usable"]]
    if window_results else []
)

consistent_windows = [
    w for w in usable_windows if w["regions_consistent"]
]

clear_peak_windows = [
    w for w in usable_windows if w["both_peaks_clear"]
]

if usable_windows:
    consistency_fraction = (
        len(consistent_windows) / len(usable_windows)
    )
    clear_peak_fraction = (
        len(clear_peak_windows) / len(usable_windows)
    )

    forehead_window_bpms = [
        w["forehead_bpm"] for w in usable_windows
    ]
    cheek_window_bpms = [
        w["cheek_bpm"] for w in usable_windows
    ]

    forehead_window_mad = median_absolute_deviation(
        forehead_window_bpms
    )
    cheek_window_mad = median_absolute_deviation(
        cheek_window_bpms
    )
else:
    consistency_fraction = 0.0
    clear_peak_fraction = 0.0
    forehead_window_mad = None
    cheek_window_mad = None

# Save diagnostics even if no windows are available.
windows_df.to_csv(WINDOW_OUTPUT, index=False)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

print("=" * 60)
print("SIGNALGUARD - PHYSIOLOGICAL SIGNAL ELIGIBILITY")
print("=" * 60)

duration = (
    float(pos_times[-1] - pos_times[0])
    if len(pos_times) > 1 else 0.0
)

print(f"\nEstimated sampling rate: {fs:.4f} Hz")
print(f"Video duration: {duration:.2f} seconds")
print(f"RGB coverage (both regions): {rgb_coverage:.1f}%")

display_metrics("Forehead - individual analysis", forehead)
display_metrics("Cheeks - individual analysis", cheeks)

print("\nShared valid interval")
print(
    "  Common valid coverage: "
    f"{np.mean(shared_valid) * 100:.1f}%"
)
print(
    "  Longest shared continuous interval: "
    f"{shared_f_metrics['longest_seconds']:.2f} seconds"
)

if shared_difference is None:
    print("  Shared-interval BPM difference: unavailable")
else:
    print(f"  Shared-interval BPM difference: {shared_difference:.1f}")

print("\nSliding-window analysis")
print(f"  Window size: {WINDOW_SECONDS:.1f} seconds")
print(f"  Step size: {WINDOW_STEP_SECONDS:.1f} seconds")
print(f"  Windows examined: {len(window_results)}")
print(f"  Usable windows: {len(usable_windows)}")

for w in window_results:
    f_bpm = w["forehead_bpm"]
    c_bpm = w["cheek_bpm"]
    diff = w["bpm_difference"]

    f_text = f"{f_bpm:.1f}" if f_bpm is not None else "N/A"
    c_text = f"{c_bpm:.1f}" if c_bpm is not None else "N/A"
    d_text = f"{diff:.1f}" if diff is not None else "N/A"

    if not w["usable"]:
        label = "UNUSABLE"
    elif w["regions_consistent"] and w["both_peaks_clear"]:
        label = "CONSISTENT"
    elif not w["regions_consistent"]:
        label = "REGION DISAGREEMENT"
    else:
        label = "PEAK QUALITY CONCERN"

    print(
        f"  {w['start_seconds']:5.1f}-{w['end_seconds']:5.1f}s | "
        f"forehead={f_text} | cheeks={c_text} | "
        f"diff={d_text} BPM | {label}"
    )

print(
    f"\nRegional agreement fraction: "
    f"{consistency_fraction * 100:.1f}%"
)
print(
    f"Windows with clear peaks in both regions: "
    f"{clear_peak_fraction * 100:.1f}%"
)

if forehead_window_mad is None:
    print("Forehead window BPM MAD: unavailable")
else:
    print(f"Forehead window BPM MAD: {forehead_window_mad:.1f}")

if cheek_window_mad is None:
    print("Cheek window BPM MAD: unavailable")
else:
    print(f"Cheek window BPM MAD: {cheek_window_mad:.1f}")

# ------------------------------------------------------------
# ELIGIBILITY CHECKS
# ------------------------------------------------------------

checks = {
    "RGB coverage sufficient":
        rgb_coverage >= MIN_COVERAGE,

    "Forehead filtered coverage sufficient":
        forehead["coverage"] >= MIN_COVERAGE,

    "Cheek filtered coverage sufficient":
        cheeks["coverage"] >= MIN_COVERAGE,

    "Forehead continuous segment sufficient":
        forehead["longest_seconds"] >= MIN_CONTINUOUS_SECONDS,

    "Cheek continuous segment sufficient":
        cheeks["longest_seconds"] >= MIN_CONTINUOUS_SECONDS,

    "Forehead spectral peak sufficient":
        forehead["peak_ratio"] is not None
        and forehead["peak_ratio"] >= 2.0,

    "Cheek spectral peak sufficient":
        cheeks["peak_ratio"] is not None
        and cheeks["peak_ratio"] >= 2.0,

    "Forehead peak quality acceptable":
        peak_is_clear(forehead),

    "Cheek peak quality acceptable":
        peak_is_clear(cheeks),

    "Shared-interval regional BPM estimates agree":
        shared_difference is not None
        and shared_difference <= MAX_REGION_HR_DIFFERENCE,

    "Enough usable sliding windows":
        len(usable_windows) >= MIN_USABLE_WINDOWS,

    "Regional agreement stable across windows":
        len(usable_windows) >= MIN_USABLE_WINDOWS
        and consistency_fraction >= MIN_CONSISTENT_WINDOW_FRACTION,

    "Peak quality acceptable across windows":
        len(usable_windows) >= MIN_USABLE_WINDOWS
        and clear_peak_fraction >= MIN_CLEAR_PEAK_WINDOW_FRACTION,

    "Forehead BPM stable across windows":
        forehead_window_mad is not None
        and forehead_window_mad <= MAX_WINDOW_BPM_MAD,

    "Cheek BPM stable across windows":
        cheek_window_mad is not None
        and cheek_window_mad <= MAX_WINDOW_BPM_MAD,
}

print("\nEligibility checks:")
for label, passed in checks.items():
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

eligible = all(checks.values())

print("\n" + "=" * 60)

if eligible:
    print("RESULT: ELIGIBLE")
    print("The video passes this prototype's signal-quality checks.")
    print("This does NOT establish authenticity.")
else:
    print("RESULT: INCONCLUSIVE")
    print("Signal quality or consistency requirements were not met.")
    print("This does NOT mean the video is fake.")

print("\nRegional difference threshold: "
      f"{MAX_REGION_HR_DIFFERENCE:.1f} BPM")
print("All quality thresholds are provisional and unvalidated.")
print("BPM values are candidate estimates, not confirmed heart rates.")
print(f"Window report saved to: {WINDOW_OUTPUT}")
print("=" * 60)
