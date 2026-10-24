"""
SignalGuard Reliability Engine & Eligibility Gate.

Provides a modular, scientifically defensible evaluation of physiological signal quality,
spectral integrity, temporal stability, regional consistency, motion contamination,
and recording conditions.

Produces an interpretable verdict:
  - ELIGIBLE: Physiological evidence meets all documented reliability criteria.
  - INCONCLUSIVE: Signal is degraded, unstable, ambiguous, or outside operating parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signalguard_utils import (
    HIGH_HZ,
    LOW_HZ,
    MAX_INTERP_GAP_SECONDS,
    MAX_INTERPOLATED_FRACTION,
    MIN_CONTINUOUS_SECONDS,
    MIN_HR_SEGMENT_SECONDS,
    MIN_LOCAL_SNR,
    MIN_MULTI_PEAK_RATIO,
    MIN_PEAK_RATIO,
    MOTION_MATCH_TOLERANCE_HZ,
    POS_WINDOW_SECONDS,
    analyse_signal,
    compute_gap_statistics,
    infer_sampling_rate,
    peak_is_clear,
)


@dataclass
class ReliabilityConfig:
    """Configurable thresholds for physiological signal reliability evaluation."""
    min_coverage_pct: float = 80.0
    min_continuous_seconds: float = 8.0
    max_interp_gap_seconds: float = 2.0
    max_interpolated_fraction: float = 0.15
    max_region_hr_difference_bpm: float = 13.0
    min_peak_ratio: float = 2.0
    min_multi_peak_ratio: float = 1.20
    min_local_snr: float = 2.0
    window_seconds: float = 10.0
    window_step_seconds: float = 2.0
    min_usable_windows: int = 3
    min_window_coverage_pct: float = 80.0
    min_consistent_window_fraction: float = 0.60
    min_clear_peak_window_fraction: float = 0.60
    max_window_bpm_mad: float = 12.0
    max_timing_jitter_pct: float = 25.0
    max_saturated_fraction: float = 0.25


def median_absolute_deviation(values: Sequence[float] | np.ndarray) -> float | None:
    """Compute robust Median Absolute Deviation (MAD) of window estimates around their median."""
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return None
    med = float(np.median(valid))
    return float(np.median(np.abs(valid - med)))


def safe_difference(a: float | None, b: float | None) -> float | None:
    """Return absolute difference between two numeric values if both exist."""
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


class SignalGuardReliabilityEngine:
    """
    Authoritative reliability engine evaluating facial rPPG signal quality across 6 categories:
      A. Signal Integrity
      B. Spectral Quality
      C. Temporal Stability
      D. Regional Consistency
      E. Motion and Tracking
      F. Recording Conditions
    """

    def __init__(self, config: ReliabilityConfig | None = None):
        self.config = config or ReliabilityConfig()

    def evaluate(
        self,
        pos_df: pd.DataFrame,
        rgb_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Evaluate physiological reliability from in-memory POS and RGB DataFrames.
        """
        cfg = self.config

        # 1. Timestamp validation & Sampling rate inference
        fs, time_details = infer_sampling_rate(pos_df, return_details=True)

        pos_times = pd.to_numeric(pos_df["time_seconds"], errors="coerce").to_numpy(dtype=float)
        rgb_times = pd.to_numeric(rgb_df["time_seconds"], errors="coerce").to_numpy(dtype=float)

        if len(pos_times) != len(rgb_times) or not np.allclose(pos_times, rgb_times, atol=1e-3, rtol=0):
            raise ValueError("POS and RGB timestamps or row counts do not match.")

        duration = float(pos_times[-1] - pos_times[0]) if len(pos_times) > 1 else 0.0

        # 2. Extract signals & tracking data
        forehead_pos = pd.to_numeric(pos_df["forehead_POS_filtered"], errors="coerce").to_numpy(float)
        cheeks_pos = pd.to_numeric(pos_df["cheeks_POS_filtered"], errors="coerce").to_numpy(float)

        motion_cols = [c for c in ["face_center_y_norm", "face_center_x_norm", "face_width_norm"] if c in rgb_df.columns]
        motion_signals = rgb_df[motion_cols].to_numpy(float) if motion_cols else None

        # Check bilateral cheek redundancy
        has_left_cheek = "left_cheek_valid" in rgb_df.columns
        has_right_cheek = "right_cheek_valid" in rgb_df.columns
        bilateral_redundancy_loss = False
        if has_left_cheek and has_right_cheek:
            left_valid_count = int(pd.to_numeric(rgb_df["left_cheek_valid"], errors="coerce").fillna(0).sum())
            right_valid_count = int(pd.to_numeric(rgb_df["right_cheek_valid"], errors="coerce").fillna(0).sum())
            if (left_valid_count == 0 and right_valid_count > 0) or (right_valid_count == 0 and left_valid_count > 0):
                bilateral_redundancy_loss = True

        # 3. Primary Regional Signal Analysis
        forehead_analysis = analyse_signal(
            forehead_pos, fs, motion_signals=motion_signals,
            max_gap_seconds=cfg.max_interp_gap_seconds, min_hr_seconds=cfg.min_continuous_seconds
        )
        cheeks_analysis = analyse_signal(
            cheeks_pos, fs, motion_signals=motion_signals,
            max_gap_seconds=cfg.max_interp_gap_seconds, min_hr_seconds=cfg.min_continuous_seconds
        )

        # Gap statistics
        f_gap = forehead_analysis.get("gap_statistics", {})
        c_gap = cheeks_analysis.get("gap_statistics", {})

        # Shared continuous interval
        shared_valid = np.isfinite(forehead_pos) & np.isfinite(cheeks_pos)
        shared_f = np.where(shared_valid, forehead_pos, np.nan)
        shared_c = np.where(shared_valid, cheeks_pos, np.nan)
        shared_f_m = analyse_signal(shared_f, fs, motion_signals=motion_signals, min_hr_seconds=cfg.min_continuous_seconds)
        shared_c_m = analyse_signal(shared_c, fs, motion_signals=motion_signals, min_hr_seconds=cfg.min_continuous_seconds)
        shared_diff = safe_difference(shared_f_m["bpm"], shared_c_m["bpm"])

        # 4. Sliding-Window Temporal Analysis
        window_size = int(round(cfg.window_seconds * fs))
        step_size = max(1, int(round(cfg.window_step_seconds * fs)))
        n_frames = len(pos_df)
        starts: list[int] = []

        if window_size > 0 and n_frames >= window_size:
            starts = list(range(0, n_frames - window_size + 1, step_size))
            final_start = n_frames - window_size
            if not starts or starts[-1] != final_start:
                starts.append(final_start)

        window_results: list[dict[str, Any]] = []
        for start in starts:
            end = start + window_size
            f_win = forehead_pos[start:end].copy()
            c_win = cheeks_pos[start:end].copy()
            common = np.isfinite(f_win) & np.isfinite(c_win)
            f_win[~common] = np.nan
            c_win[~common] = np.nan

            common_cov = float(np.mean(common) * 100.0)
            win_motion = motion_signals[start:end] if motion_signals is not None else None

            f_win_m = analyse_signal(f_win, fs, motion_signals=win_motion, min_hr_seconds=cfg.min_continuous_seconds)
            c_win_m = analyse_signal(c_win, fs, motion_signals=win_motion, min_hr_seconds=cfg.min_continuous_seconds)

            both_avail = (f_win_m["bpm"] is not None and c_win_m["bpm"] is not None)
            diff = safe_difference(f_win_m["bpm"], c_win_m["bpm"])

            usable = (
                common_cov >= cfg.min_window_coverage_pct
                and f_win_m["longest_seconds"] >= cfg.min_continuous_seconds
                and c_win_m["longest_seconds"] >= cfg.min_continuous_seconds
                and both_avail
            )

            both_clear = peak_is_clear(f_win_m) and peak_is_clear(c_win_m)
            consistent = usable and diff is not None and diff <= cfg.max_region_hr_difference_bpm

            window_results.append({
                "start_seconds": float(pos_times[start]),
                "end_seconds": float(pos_times[end - 1]),
                "common_coverage_pct": common_cov,
                "forehead_bpm": f_win_m["bpm"],
                "cheek_bpm": c_win_m["bpm"],
                "bpm_difference": diff,
                "forehead_peak_ratio": f_win_m["peak_ratio"],
                "cheek_peak_ratio": c_win_m["peak_ratio"],
                "forehead_local_snr": f_win_m.get("local_snr"),
                "cheek_local_snr": c_win_m.get("local_snr"),
                "forehead_multi_peak_ratio": f_win_m["multi_peak_ratio"],
                "cheek_multi_peak_ratio": c_win_m["multi_peak_ratio"],
                "usable": usable,
                "both_peaks_clear": both_clear,
                "regions_consistent": consistent,
            })

        usable_windows = [w for w in window_results if w["usable"]]
        consistent_windows = [w for w in usable_windows if w["regions_consistent"]]
        clear_windows = [w for w in usable_windows if w["both_peaks_clear"]]

        consistency_fraction = (len(consistent_windows) / len(usable_windows)) if usable_windows else 0.0
        clear_fraction = (len(clear_windows) / len(usable_windows)) if usable_windows else 0.0

        f_mad = median_absolute_deviation([w["forehead_bpm"] for w in usable_windows])
        c_mad = median_absolute_deviation([w["cheek_bpm"] for w in usable_windows])

        # 5. Recording Conditions
        mean_brightness = float(rgb_df["cheeks_brightness"].mean()) if "cheeks_brightness" in rgb_df.columns else np.nan
        mean_saturation = float(rgb_df["cheeks_saturated_fraction"].mean()) if "cheeks_saturated_fraction" in rgb_df.columns else 0.0

        # 6. Evaluation Categories Compilation
        categories = {
            "A_signal_integrity": {
                "forehead_valid_fraction": f_gap.get("valid_fraction", 0.0),
                "cheeks_valid_fraction": c_gap.get("valid_fraction", 0.0),
                "forehead_interpolated_fraction": f_gap.get("interpolated_fraction_potential", 0.0),
                "cheeks_interpolated_fraction": c_gap.get("interpolated_fraction_potential", 0.0),
                "forehead_continuous_seconds": forehead_analysis["longest_seconds"],
                "cheeks_continuous_seconds": cheeks_analysis["longest_seconds"],
                "usable_segments_count": max(forehead_analysis.get("usable_segments_count", 0), cheeks_analysis.get("usable_segments_count", 0)),
            },
            "B_spectral_quality": {
                "forehead_bpm": forehead_analysis["bpm"],
                "cheeks_bpm": cheeks_analysis["bpm"],
                "forehead_peak_ratio": forehead_analysis["peak_ratio"],
                "cheeks_peak_ratio": cheeks_analysis["peak_ratio"],
                "forehead_local_snr": forehead_analysis.get("local_snr"),
                "cheeks_local_snr": cheeks_analysis.get("local_snr"),
                "forehead_prominence": forehead_analysis.get("prominence"),
                "cheeks_prominence": cheeks_analysis.get("prominence"),
                "forehead_multi_peak_ratio": forehead_analysis["multi_peak_ratio"],
                "cheeks_multi_peak_ratio": cheeks_analysis["multi_peak_ratio"],
                "forehead_uncertain": forehead_analysis.get("uncertain", True),
                "cheeks_uncertain": cheeks_analysis.get("uncertain", True),
            },
            "C_temporal_stability": {
                "total_windows": len(window_results),
                "usable_windows": len(usable_windows),
                "consistent_window_fraction": consistency_fraction,
                "clear_peak_window_fraction": clear_fraction,
                "forehead_bpm_mad": f_mad,
                "cheeks_bpm_mad": c_mad,
            },
            "D_regional_consistency": {
                "shared_interval_bpm_difference": shared_diff,
                "bilateral_redundancy_loss": bilateral_redundancy_loss,
            },
            "E_motion_and_tracking": {
                "motion_frequencies_hz": forehead_analysis.get("motion_frequencies_hz", []),
                "forehead_motion_overlap_risk": forehead_analysis.get("motion_overlap_risk", "NONE"),
                "cheeks_motion_overlap_risk": cheeks_analysis.get("motion_overlap_risk", "NONE"),
                "forehead_is_motion_contaminated": forehead_analysis.get("is_motion_contaminated", False),
                "cheeks_is_motion_contaminated": cheeks_analysis.get("is_motion_contaminated", False),
            },
            "F_recording_conditions": {
                "mean_brightness": mean_brightness,
                "mean_saturation_fraction": mean_saturation,
                "timing_jitter_pct": time_details["timing_jitter_pct"],
                "sampling_rate_hz": fs,
                "total_duration_sec": duration,
            },
        }

        # 7. Explicit Hard Safety Checks
        checks: dict[str, dict[str, Any]] = {
            "forehead_coverage": {
                "name": "Forehead coverage sufficient",
                "passed": forehead_analysis["coverage"] >= cfg.min_coverage_pct,
                "observed": forehead_analysis["coverage"],
                "threshold": f">= {cfg.min_coverage_pct}%",
            },
            "cheeks_coverage": {
                "name": "Cheeks coverage sufficient",
                "passed": cheeks_analysis["coverage"] >= cfg.min_coverage_pct,
                "observed": cheeks_analysis["coverage"],
                "threshold": f">= {cfg.min_coverage_pct}%",
            },
            "forehead_continuous_duration": {
                "name": "Forehead continuous duration sufficient",
                "passed": forehead_analysis["longest_seconds"] >= cfg.min_continuous_seconds,
                "observed": forehead_analysis["longest_seconds"],
                "threshold": f">= {cfg.min_continuous_seconds}s",
            },
            "cheeks_continuous_duration": {
                "name": "Cheeks continuous duration sufficient",
                "passed": cheeks_analysis["longest_seconds"] >= cfg.min_continuous_seconds,
                "observed": cheeks_analysis["longest_seconds"],
                "threshold": f">= {cfg.min_continuous_seconds}s",
            },
            "forehead_spectral_peak": {
                "name": "Forehead spectral peak clear",
                "passed": peak_is_clear(forehead_analysis),
                "observed": (
                    f"peak={forehead_analysis['peak_ratio']:.2f}, comp={forehead_analysis['multi_peak_ratio']:.2f}"
                    if forehead_analysis["multi_peak_ratio"] is not None
                    else f"peak={forehead_analysis['peak_ratio']:.2f}"
                ),
                "threshold": f"peak >= {cfg.min_peak_ratio}, comp >= {cfg.min_multi_peak_ratio}",
            },
            "cheeks_spectral_peak": {
                "name": "Cheeks spectral peak clear",
                "passed": peak_is_clear(cheeks_analysis),
                "observed": (
                    f"peak={cheeks_analysis['peak_ratio']:.2f}, comp={cheeks_analysis['multi_peak_ratio']:.2f}"
                    if cheeks_analysis["multi_peak_ratio"] is not None
                    else f"peak={cheeks_analysis['peak_ratio']:.2f}"
                ),
                "threshold": f"peak >= {cfg.min_peak_ratio}, comp >= {cfg.min_multi_peak_ratio}",
            },
            "regional_bpm_agreement": {
                "name": "Regional BPM estimates agree on shared interval",
                "passed": shared_diff is not None and shared_diff <= cfg.max_region_hr_difference_bpm,
                "observed": shared_diff,
                "threshold": f"<= {cfg.max_region_hr_difference_bpm} BPM",
            },
            "sliding_windows_count": {
                "name": "Sufficient usable sliding windows",
                "passed": len(usable_windows) >= cfg.min_usable_windows,
                "observed": len(usable_windows),
                "threshold": f">= {cfg.min_usable_windows}",
            },
            "window_regional_consistency": {
                "name": "Regional consistency stable across windows",
                "passed": len(usable_windows) >= cfg.min_usable_windows and consistency_fraction >= cfg.min_consistent_window_fraction,
                "observed": consistency_fraction,
                "threshold": f">= {cfg.min_consistent_window_fraction * 100}%",
            },
            "window_peak_quality": {
                "name": "Peak quality stable across windows",
                "passed": len(usable_windows) >= cfg.min_usable_windows and clear_fraction >= cfg.min_clear_peak_window_fraction,
                "observed": clear_fraction,
                "threshold": f">= {cfg.min_clear_peak_window_fraction * 100}%",
            },
            "forehead_bpm_mad": {
                "name": "Forehead BPM temporally stable (MAD)",
                "passed": f_mad is not None and f_mad <= cfg.max_window_bpm_mad,
                "observed": f_mad,
                "threshold": f"<= {cfg.max_window_bpm_mad} BPM",
            },
            "cheeks_bpm_mad": {
                "name": "Cheeks BPM temporally stable (MAD)",
                "passed": c_mad is not None and c_mad <= cfg.max_window_bpm_mad,
                "observed": c_mad,
                "threshold": f"<= {cfg.max_window_bpm_mad} BPM",
            },
            "interpolation_limit": {
                "name": "Interpolated fraction within limit",
                "passed": (
                    f_gap.get("interpolated_fraction_potential", 0.0) <= cfg.max_interpolated_fraction
                    and c_gap.get("interpolated_fraction_potential", 0.0) <= cfg.max_interpolated_fraction
                ),
                "observed": max(f_gap.get("interpolated_fraction_potential", 0.0), c_gap.get("interpolated_fraction_potential", 0.0)),
                "threshold": f"<= {cfg.max_interpolated_fraction * 100}%",
            },
        }

        all_passed = all(chk["passed"] for chk in checks.values())
        verdict = "ELIGIBLE" if all_passed else "INCONCLUSIVE"

        # Explicit failure reasons and warnings
        reasons: list[str] = []
        for key, chk in checks.items():
            if not chk["passed"]:
                obs_str = f"{chk['observed']:.2f}" if isinstance(chk["observed"], (int, float)) else str(chk["observed"])
                reasons.append(f"{chk['name']} (Observed: {obs_str}, Expected: {chk['threshold']})")

        warnings: list[str] = []
        if bilateral_redundancy_loss:
            warnings.append("Loss of bilateral cheek redundancy: one cheek ROI was completely unavailable (single cheek fallback active).")
        if forehead_analysis.get("motion_overlap_risk") in {"MODERATE", "HIGH"}:
            warnings.append(f"Forehead pulse candidate in proximity to head tracking motion frequency ({forehead_analysis.get('motion_overlap_risk')} risk).")
        if cheeks_analysis.get("motion_overlap_risk") in {"MODERATE", "HIGH"}:
            warnings.append(f"Cheeks pulse candidate in proximity to head tracking motion frequency ({cheeks_analysis.get('motion_overlap_risk')} risk).")
        if time_details["timing_jitter_pct"] > 15.0:
            warnings.append(f"Elevated frame timing jitter detected ({time_details['timing_jitter_pct']:.1f}%).")

        suggestions: list[str] = []
        if not checks["regional_bpm_agreement"]["passed"]:
            suggestions.append("Check for differential motion or illumination gradients across facial ROIs causing regional peak divergence.")
        if not checks["forehead_continuous_duration"]["passed"] or not checks["cheeks_continuous_duration"]["passed"]:
            suggestions.append("Ensure uninterrupted face tracking for at least 8.0 continuous seconds without head turnouts or occlusions.")
        if not checks["sliding_windows_count"]["passed"]:
            suggestions.append(f"Provide video duration of at least {cfg.window_seconds + (cfg.min_usable_windows - 1) * cfg.window_step_seconds:.1f} seconds to populate sliding windows.")

        return {
            "verdict": verdict,
            "sampling_rate_hz": fs,
            "duration_seconds": duration,
            "forehead_candidate_bpm": forehead_analysis["bpm"],
            "cheeks_candidate_bpm": cheeks_analysis["bpm"],
            "regional_bpm_difference": shared_diff,
            "checks": checks,
            "categories": categories,
            "failure_reasons": reasons,
            "warnings": warnings,
            "suggested_improvements": suggestions,
            "window_results": window_results,
        }

    def evaluate_files(
        self,
        pos_path: str | Path,
        rgb_path: str | Path,
        window_output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load CSV files from disk and evaluate reliability."""
        pos_p = Path(pos_path)
        rgb_p = Path(rgb_path)

        if not pos_p.is_file():
            raise FileNotFoundError(f"POS signals file not found: {pos_p}")
        if not rgb_p.is_file():
            raise FileNotFoundError(f"RGB signals file not found: {rgb_p}")

        pos_df = pd.read_csv(pos_p)
        rgb_df = pd.read_csv(rgb_p)

        report = self.evaluate(pos_df, rgb_df)

        if window_output_path:
            win_p = Path(window_output_path)
            win_p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(report["window_results"]).to_csv(win_p, index=False)

        return report

    @staticmethod
    def format_report(report: dict[str, Any]) -> str:
        """Format an evaluation report for terminal presentation."""
        lines = [
            "=" * 70,
            "SIGNALGUARD PHYSIOLOGICAL SIGNAL RELIABILITY ASSESSMENT",
            "=" * 70,
            f"Sampling Rate:        {report['sampling_rate_hz']:.4f} Hz",
            f"Video Duration:       {report['duration_seconds']:.2f} seconds",
            f"Forehead Candidate:   {report['forehead_candidate_bpm']:.1f} BPM" if report['forehead_candidate_bpm'] else "Forehead Candidate:   N/A",
            f"Cheeks Candidate:     {report['cheeks_candidate_bpm']:.1f} BPM" if report['cheeks_candidate_bpm'] else "Cheeks Candidate:     N/A",
            f"Regional Difference:  {report['regional_bpm_difference']:.1f} BPM" if report['regional_bpm_difference'] is not None else "Regional Difference:  N/A",
            "",
            "RELIABILITY CHECKS:",
        ]

        for _, chk in report["checks"].items():
            status = "PASS" if chk["passed"] else "FAIL"
            obs_val = f"{chk['observed']:.2f}" if isinstance(chk["observed"], (int, float)) else str(chk["observed"])
            lines.append(f"  [{status:<4}] {chk['name']:<46} (Obs: {obs_val:<6} | Exp: {chk['threshold']})")

        lines.extend([
            "",
            "-" * 70,
            f"FINAL VERDICT: {report['verdict']}",
            "-" * 70,
        ])

        if report["verdict"] == "ELIGIBLE":
            lines.append("Physiological signals meet all stability and regional agreement criteria.")
            lines.append("Notice: ELIGIBILITY does NOT confirm video authenticity (not a deepfake verdict).")
        else:
            lines.append("Signal quality, temporal continuity, or regional agreement criteria failed.")
            lines.append("Notice: INCONCLUSIVE does NOT imply manipulation or deepfake fabrication.")
            if report["failure_reasons"]:
                lines.append("\nReasons for Abstention:")
                for r in report["failure_reasons"]:
                    lines.append(f"  - {r}")

        if report["warnings"]:
            lines.append("\nWarnings:")
            for w in report["warnings"]:
                lines.append(f"  ! {w}")

        if report["suggested_improvements"]:
            lines.append("\nSuggested Recording Adjustments:")
            for s in report["suggested_improvements"]:
                lines.append(f"  * {s}")

        lines.append("=" * 70)
        return "\n".join(lines)


# Standalone backward-compatible CLI execution
def main():
    parser = argparse.ArgumentParser(description="SignalGuard Reliability Engine & Eligibility Gate")
    parser.add_argument("--pos", type=str, default="pos_signals.csv", help="Path to pos_signals.csv")
    parser.add_argument("--rgb", type=str, default="rgb_signals.csv", help="Path to rgb_signals.csv")
    parser.add_argument("--out-window", type=str, default="window_analysis.csv", help="Output path for window_analysis.csv")
    parser.add_argument("--out-json", type=str, default=None, help="Optional output path for reliability report JSON")

    args = parser.parse_args()

    pos_file = Path(args.pos)
    rgb_file = Path(args.rgb)

    if not pos_file.is_file() or not rgb_file.is_file():
        print(f"ERROR: Required inputs missing: {pos_file} or {rgb_file}")
        print("Run signal_extractor.py and pos_extractor.py before the eligibility gate.")
        return 1

    engine = SignalGuardReliabilityEngine()
    try:
        report = engine.evaluate_files(pos_file, rgb_file, window_output_path=args.out_window)
    except Exception as exc:
        print(f"ERROR during reliability evaluation: {exc}")
        return 1

    formatted_text = engine.format_report(report)
    print(formatted_text)

    if args.out_json:
        out_j = Path(args.out_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        # Convert non-serializable fields
        serializable_report = json.loads(json.dumps(report, default=str))
        with out_j.open("w", encoding="utf-8") as f:
            json.dump(serializable_report, f, indent=2)
        print(f"\nSaved structured reliability report to: {out_j}")

    return 0 if report["verdict"] == "ELIGIBLE" else 0


if __name__ == "__main__":
    sys.exit(main())
