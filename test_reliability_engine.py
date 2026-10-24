"""
SignalGuard Phase 2: Comprehensive Synthetic Reliability Engine Test Suite.

Covers all 15 deterministic synthetic test scenarios specified in Phase 4:
  1. Clean synthetic pulse at known frequency.
  2. Pulse signal with Gaussian noise.
  3. Pulse mixed with a stronger motion-related component.
  4. Signal containing two competing spectral peaks.
  5. Short missing-data gap (interpolated within time limit).
  6. Long missing-data gap (uninterpolated, strictly segmented).
  7. Missing-data gap at the beginning of recording (no boundary extrapolation).
  8. Nearly constant signal (zero-variance safeguard).
  9. Empty or fully invalid signal (graceful failure).
  10. Insufficient-duration signal (< 8.0s).
  11. Signal with irregular timestamps or dropped samples.
  12. Signal with different known frequencies in separate windows (temporal instability).
  13. Region-specific missing data where one cheek remains valid (redundancy loss visible).
  14. Case where movement and pulse frequencies overlap (uncertainty and risk reported).
  15. Case where regional estimates disagree (INCONCLUSIVE verdict).

All tests use fixed random seeds and verify actual output numbers, tolerances, and verdicts.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

import signalguard_utils as utils
from eligibility_gate import ReliabilityConfig, SignalGuardReliabilityEngine


def generate_synthetic_channels(
    duration_sec: float,
    fs: float,
    pulse_bpm: float,
    pulse_amp: float = 0.015,
    noise_std: float = 0.0,
    base_rgb: tuple[float, float, float] = (140.0, 115.0, 95.0),
    motion_bpm: float | None = None,
    motion_amp: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic skin RGB time series based on optical photoplethysmography principles:
    blood volume pulse (BVP) produces strongest modulation in the Green channel (hemoglobin absorption peak).
    """
    rng = np.random.RandomState(seed)
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs

    r0, g0, b0 = base_rgb
    f_pulse = pulse_bpm / 60.0
    pulse_wave = np.sin(2.0 * np.pi * f_pulse * t)

    # In rPPG: Green has negative PPG variation, while Red/Blue have smaller variation
    g_sig = g0 * (1.0 + pulse_amp * pulse_wave)
    r_sig = r0 * (1.0 - 0.3 * pulse_amp * pulse_wave)
    b_sig = b0 * (1.0 - 0.2 * pulse_amp * pulse_wave)

    if motion_bpm is not None and motion_amp > 0:
        f_motion = motion_bpm / 60.0
        motion_wave = np.sin(2.0 * np.pi * f_motion * t)
        # Realistic differential illumination/shadowing across channels
        r_sig += r0 * motion_amp * 1.1 * motion_wave
        g_sig += g0 * motion_amp * 0.9 * motion_wave
        b_sig += b0 * motion_amp * 1.0 * motion_wave

    if noise_std > 0:
        r_sig += rng.normal(0, noise_std * r0, n)
        g_sig += rng.normal(0, noise_std * g0, n)
        b_sig += rng.normal(0, noise_std * b0, n)

    rgb = np.column_stack([r_sig, g_sig, b_sig])
    return t, rgb


def build_pipeline_dataframes(
    t: np.ndarray,
    f_rgb: np.ndarray,
    c_rgb: np.ndarray,
    fs: float,
    motion_series: np.ndarray | None = None,
    left_cheek_valid: bool = True,
    right_cheek_valid: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Helper to assemble pos_df and rgb_df matching pipeline schema."""
    n = len(t)
    f_raw = utils.pos_from_rgb(f_rgb, fs)
    c_raw = utils.pos_from_rgb(c_rgb, fs)

    f_filt = utils.filter_pos_signal(f_raw, fs)
    c_filt = utils.filter_pos_signal(c_raw, fs)

    pos_df = pd.DataFrame({
        "time_seconds": t,
        "forehead_POS": f_raw,
        "forehead_POS_filtered": f_filt,
        "cheeks_POS": c_raw,
        "cheeks_POS_filtered": c_filt,
    })

    rgb_dict = {
        "time_seconds": t,
        "forehead_R": f_rgb[:, 0],
        "forehead_G": f_rgb[:, 1],
        "forehead_B": f_rgb[:, 2],
        "cheeks_R": c_rgb[:, 0],
        "cheeks_G": c_rgb[:, 1],
        "cheeks_B": c_rgb[:, 2],
        "left_cheek_valid": np.full(n, 1 if left_cheek_valid else 0),
        "right_cheek_valid": np.full(n, 1 if right_cheek_valid else 0),
        "cheeks_brightness": np.mean(c_rgb, axis=1),
        "cheeks_saturated_fraction": np.zeros(n),
    }

    if motion_series is not None:
        rgb_dict["face_center_y_norm"] = motion_series
        rgb_dict["face_center_x_norm"] = np.zeros(n)
        rgb_dict["face_width_norm"] = np.full(n, 0.4)

    rgb_df = pd.DataFrame(rgb_dict)
    return pos_df, rgb_df


class TestSignalGuardReliabilityEngine(unittest.TestCase):
    """Rigorous test suite covering the 15 required synthetic scenarios."""

    def setUp(self):
        self.fs = 30.0
        self.engine = SignalGuardReliabilityEngine()

    # ------------------------------------------------------------
    # 1. Clean synthetic pulse at known frequency
    # ------------------------------------------------------------
    def test_01_clean_synthetic_pulse(self):
        target_bpm = 75.0
        duration = 20.0
        t, rgb = generate_synthetic_channels(duration, self.fs, target_bpm, pulse_amp=0.02, noise_std=0.0)

        raw_pos = utils.pos_from_rgb(rgb, self.fs)
        filt_pos = utils.filter_pos_signal(raw_pos, self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs)

        self.assertIsNotNone(metrics["bpm"])
        error_bpm = abs(metrics["bpm"] - target_bpm)
        self.assertLess(error_bpm, 1.5, f"BPM error {error_bpm:.2f} exceeded tolerance 1.5 BPM")
        self.assertTrue(utils.peak_is_clear(metrics))
        self.assertGreater(metrics["peak_ratio"], 3.0)

    # ------------------------------------------------------------
    # 2. Pulse signal with noise
    # ------------------------------------------------------------
    def test_02_pulse_with_noise(self):
        target_bpm = 78.0
        duration = 20.0
        t, rgb = generate_synthetic_channels(duration, self.fs, target_bpm, pulse_amp=0.015, noise_std=0.0025, seed=101)

        raw_pos = utils.pos_from_rgb(rgb, self.fs)
        filt_pos = utils.filter_pos_signal(raw_pos, self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs)

        self.assertIsNotNone(metrics["bpm"])
        error_bpm = abs(metrics["bpm"] - target_bpm)
        self.assertLess(error_bpm, 2.0, f"Noisy BPM error {error_bpm:.2f} exceeded tolerance 2.0 BPM")
        self.assertGreater(metrics["local_snr"], 2.0, "Local spectral SNR should remain above 2.0 for identifiable pulse")

    # ------------------------------------------------------------
    # 3. Pulse mixed with a stronger motion-related component
    # ------------------------------------------------------------
    def test_03_pulse_mixed_with_strong_motion(self):
        pulse_bpm = 72.0
        motion_bpm = 130.0
        duration = 25.0

        t, rgb = generate_synthetic_channels(
            duration, self.fs, pulse_bpm=pulse_bpm, pulse_amp=0.012,
            motion_bpm=motion_bpm, motion_amp=0.028, noise_std=0.001, seed=202
        )
        tracking_motion = np.sin(2.0 * np.pi * (motion_bpm / 60.0) * t)

        raw_pos = utils.pos_from_rgb(rgb, self.fs)
        filt_pos = utils.filter_pos_signal(raw_pos, self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs, motion_signals=tracking_motion)

        # Candidate peaks must contain both frequencies
        cand_bpms = [c["bpm"] for c in metrics["candidate_peaks"]]
        has_pulse_cand = any(abs(b - pulse_bpm) < 2.5 for b in cand_bpms)
        has_motion_cand = any(abs(b - motion_bpm) < 2.5 for b in cand_bpms)

        self.assertTrue(has_pulse_cand, f"Pulse candidate ~72 BPM missing from {cand_bpms}")
        self.assertTrue(has_motion_cand, f"Motion candidate ~130 BPM missing from {cand_bpms}")
        # The motion peak should be flagged as motion
        motion_flags = [c["is_motion"] for c in metrics["candidate_peaks"] if abs(c["bpm"] - motion_bpm) < 3.0]
        self.assertTrue(any(motion_flags), "Motion peak was not flagged as motion")

    # ------------------------------------------------------------
    # 4. Signal containing two competing spectral peaks
    # ------------------------------------------------------------
    def test_04_competing_spectral_peaks(self):
        duration = 20.0
        t = np.arange(int(round(duration * self.fs))) / self.fs
        # Mix two distinct physiological-range pulses without motion
        f1, f2 = 70.0 / 60.0, 100.0 / 60.0
        wave = 0.015 * np.sin(2 * np.pi * f1 * t) + 0.014 * np.sin(2 * np.pi * f2 * t)
        rgb = np.column_stack([
            140.0 * (1 - 0.2 * wave),
            115.0 * (1 + wave),
            95.0 * (1 - 0.2 * wave),
        ])

        filt_pos = utils.filter_pos_signal(utils.pos_from_rgb(rgb, self.fs), self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs)

        self.assertTrue(metrics["has_competing_peak"], "Competing peak was not flagged")
        self.assertIsNotNone(metrics["multi_peak_ratio"])
        # Multi-peak ratio should be low (competing closely in power)
        self.assertLess(metrics["multi_peak_ratio"], 1.25)
        self.assertTrue(metrics["uncertain"], "Signal with closely competing peaks must be flagged uncertain")

    # ------------------------------------------------------------
    # 5. Short missing-data gap (interpolated within limit)
    # ------------------------------------------------------------
    def test_05_short_missing_data_gap(self):
        duration = 20.0
        target_bpm = 74.0
        t, rgb = generate_synthetic_channels(duration, self.fs, target_bpm, seed=303)

        # Introduce a 1.0s gap in the middle (seconds 10.0 to 11.0)
        gap_start, gap_end = int(10.0 * self.fs), int(11.0 * self.fs)
        rgb[gap_start:gap_end, :] = np.nan

        # Test interpolation function directly with mask return
        interp_rgb, raw_valid, interp_mask = utils.interpolate_invalid_samples(
            rgb, fs=self.fs, max_gap_seconds=2.0, return_mask=True
        )

        self.assertEqual(np.sum(~raw_valid), gap_end - gap_start)
        self.assertEqual(np.sum(interp_mask), gap_end - gap_start)
        self.assertTrue(np.all(np.isfinite(interp_rgb)), "Short gap should be fully interpolated")

        # Verify pipeline recovers accurate BPM through the bridged gap
        filt_pos = utils.filter_pos_signal(utils.pos_from_rgb(rgb, self.fs), self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs)
        self.assertIsNotNone(metrics["bpm"])
        self.assertLess(abs(metrics["bpm"] - target_bpm), 2.0)

    # ------------------------------------------------------------
    # 6. Long missing-data gap (uninterpolated, segmented)
    # ------------------------------------------------------------
    def test_06_long_missing_data_gap(self):
        duration = 25.0
        target_bpm = 76.0
        t, rgb = generate_synthetic_channels(duration, self.fs, target_bpm, seed=404)

        # Introduce a 5.0s gap (exceeding MAX_INTERP_GAP_SECONDS = 2.0s)
        gap_start, gap_end = int(10.0 * self.fs), int(15.0 * self.fs)
        rgb[gap_start:gap_end, :] = np.nan

        interp_rgb, raw_valid, interp_mask = utils.interpolate_invalid_samples(
            rgb, fs=self.fs, max_gap_seconds=2.0, return_mask=True
        )

        # The 5s gap must NOT be interpolated
        self.assertEqual(np.sum(interp_mask), 0, "Gaps > 2.0s must never be interpolated")
        self.assertTrue(np.all(np.isnan(interp_rgb[gap_start:gap_end])), "Long gap must remain NaN")

        gap_stats = utils.compute_gap_statistics(rgb, self.fs, max_gap_seconds=2.0)
        self.assertEqual(gap_stats["usable_segments_count"], 2, "Should identify 2 separate continuous segments")

    # ------------------------------------------------------------
    # 7. Missing-data gap at the beginning of recording
    # ------------------------------------------------------------
    def test_07_gap_at_beginning_not_extrapolated(self):
        duration = 20.0
        target_bpm = 80.0
        t, rgb = generate_synthetic_channels(duration, self.fs, target_bpm, seed=505)

        # First 3.0s are NaN
        rgb[: int(3.0 * self.fs), :] = np.nan

        interp_rgb, raw_valid, interp_mask = utils.interpolate_invalid_samples(
            rgb, fs=self.fs, max_gap_seconds=2.0, return_mask=True
        )

        # Boundary gaps must NEVER be extrapolated
        self.assertEqual(np.sum(interp_mask), 0, "Boundary gaps at start must never be extrapolated")
        self.assertTrue(np.all(np.isnan(interp_rgb[: int(3.0 * self.fs)])))

        # Remaining 17.0s should still be analyzed
        filt_pos = utils.filter_pos_signal(utils.pos_from_rgb(rgb, self.fs), self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs)
        self.assertIsNotNone(metrics["bpm"])
        self.assertLess(abs(metrics["bpm"] - target_bpm), 2.0)

    # ------------------------------------------------------------
    # 8. Nearly constant signal
    # ------------------------------------------------------------
    def test_08_nearly_constant_signal(self):
        n = int(round(15.0 * self.fs))
        rgb = np.full((n, 3), 120.0)

        raw_pos = utils.pos_from_rgb(rgb, self.fs)
        filt_pos = utils.filter_pos_signal(raw_pos, self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs)

        self.assertIsNone(metrics["bpm"], "Constant signal must yield None BPM")

    # ------------------------------------------------------------
    # 9. Empty or fully invalid signal
    # ------------------------------------------------------------
    def test_09_empty_or_fully_invalid_signal(self):
        # Test empty array
        empty_rgb = np.zeros((0, 3))
        raw_pos = utils.pos_from_rgb(empty_rgb, self.fs)
        self.assertEqual(len(raw_pos), 0)
        metrics_empty = utils.analyse_signal(raw_pos, self.fs)
        self.assertIsNone(metrics_empty["bpm"])

        # Test all-NaN array
        nan_rgb = np.full((300, 3), np.nan)
        filt_nan = utils.filter_pos_signal(utils.pos_from_rgb(nan_rgb, self.fs), self.fs)
        metrics_nan = utils.analyse_signal(filt_nan, self.fs)
        self.assertIsNone(metrics_nan["bpm"])
        self.assertEqual(metrics_nan["coverage"], 0.0)

    # ------------------------------------------------------------
    # 10. Insufficient-duration signal
    # ------------------------------------------------------------
    def test_10_insufficient_duration_signal(self):
        # 4.0 seconds duration (< MIN_HR_SEGMENT_SECONDS = 8.0s)
        duration = 4.0
        t, rgb = generate_synthetic_channels(duration, self.fs, 75.0, seed=606)

        filt_pos = utils.filter_pos_signal(utils.pos_from_rgb(rgb, self.fs), self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs, min_hr_seconds=8.0)

        self.assertIsNone(metrics["bpm"])
        self.assertIn("minimum required", metrics["selection_reason"])

    # ------------------------------------------------------------
    # 11. Signal with irregular timestamps or dropped samples
    # ------------------------------------------------------------
    def test_11_irregular_timestamps_jitter(self):
        duration = 20.0
        target_bpm = 77.0
        nominal_dt = 1.0 / self.fs
        n = int(round(duration * self.fs))

        # Add 12% random timing jitter to timestamps
        rng = np.random.RandomState(707)
        jitter = rng.uniform(-0.12 * nominal_dt, 0.12 * nominal_dt, n)
        jitter[0] = 0
        t = np.cumsum(np.full(n, nominal_dt) + jitter)

        df = pd.DataFrame({"time_seconds": t})
        inferred_fs, details = utils.infer_sampling_rate(df, return_details=True)

        self.assertAlmostEqual(inferred_fs, self.fs, delta=0.5)
        self.assertGreater(details["timing_jitter_pct"], 0.0)

        _, rgb = generate_synthetic_channels(duration, inferred_fs, target_bpm, seed=707)
        filt_pos = utils.filter_pos_signal(utils.pos_from_rgb(rgb, inferred_fs), inferred_fs)
        metrics = utils.analyse_signal(filt_pos, inferred_fs)
        self.assertIsNotNone(metrics["bpm"])
        self.assertLess(abs(metrics["bpm"] - target_bpm), 2.0)

    # ------------------------------------------------------------
    # 12. Signal with different known frequencies in separate windows
    # ------------------------------------------------------------
    def test_12_frequency_shift_across_windows(self):
        # Linearly ramping physiological frequency from 55 to 125 BPM over 30s
        duration = 30.0
        n = int(round(duration * self.fs))
        t = np.arange(n) / self.fs
        f_start = 55.0 / 60.0
        f_end = 125.0 / 60.0
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t**2) / duration)
        pulse = 0.015 * np.sin(phase)
        rgb = np.column_stack([140.0 * (1 - 0.2 * pulse), 115.0 * (1 + pulse), 95.0 * (1 - 0.2 * pulse)])

        pos_df, rgb_df = build_pipeline_dataframes(t, rgb, rgb, self.fs)
        report = self.engine.evaluate(pos_df, rgb_df)

        # Sliding window analysis must detect high BPM spread (MAD)
        f_mad = report["categories"]["C_temporal_stability"]["forehead_bpm_mad"]
        self.assertIsNotNone(f_mad)
        self.assertGreater(f_mad, 12.0, f"Expected high MAD from frequency shift, got {f_mad}")
        self.assertFalse(report["checks"]["forehead_bpm_mad"]["passed"])
        self.assertEqual(report["verdict"], "INCONCLUSIVE")

    # ------------------------------------------------------------
    # 13. Region-specific missing data where one cheek remains valid
    # ------------------------------------------------------------
    def test_13_region_specific_missing_data_redundancy_loss(self):
        duration = 20.0
        target_bpm = 74.0
        t, rgb = generate_synthetic_channels(duration, self.fs, target_bpm, seed=901)

        # Left cheek valid, Right cheek completely invalid (simulating yaw turn)
        pos_df, rgb_df = build_pipeline_dataframes(t, rgb, rgb, self.fs, left_cheek_valid=True, right_cheek_valid=False)

        report = self.engine.evaluate(pos_df, rgb_df)
        self.assertTrue(report["categories"]["D_regional_consistency"]["bilateral_redundancy_loss"])
        self.assertTrue(any("redundancy" in w.lower() for w in report["warnings"]))

    # ------------------------------------------------------------
    # 14. Case where movement and pulse frequencies overlap
    # ------------------------------------------------------------
    def test_14_motion_pulse_frequency_overlap(self):
        duration = 25.0
        overlap_bpm = 75.0
        f_hz = overlap_bpm / 60.0

        t, rgb = generate_synthetic_channels(
            duration, self.fs, pulse_bpm=overlap_bpm, pulse_amp=0.015,
            motion_bpm=overlap_bpm, motion_amp=0.030, seed=902
        )
        motion_series = np.sin(2.0 * np.pi * f_hz * t)

        filt_pos = utils.filter_pos_signal(utils.pos_from_rgb(rgb, self.fs), self.fs)
        metrics = utils.analyse_signal(filt_pos, self.fs, motion_signals=motion_series)

        # When motion and pulse have identical frequencies, overlap risk must be HIGH
        self.assertEqual(metrics["motion_overlap_risk"], "HIGH")
        self.assertTrue(metrics["uncertain"], "Overlapping motion and pulse must be flagged uncertain")

    # ------------------------------------------------------------
    # 15. Case where regional estimates disagree
    # ------------------------------------------------------------
    def test_15_regional_estimates_disagree_inconclusive(self):
        duration = 20.0
        # Forehead at 70 BPM, Cheeks at 115 BPM
        t, f_rgb = generate_synthetic_channels(duration, self.fs, 70.0, seed=903)
        _, c_rgb = generate_synthetic_channels(duration, self.fs, 115.0, seed=904)

        pos_df, rgb_df = build_pipeline_dataframes(t, f_rgb, c_rgb, self.fs)
        report = self.engine.evaluate(pos_df, rgb_df)

        diff = report["regional_bpm_difference"]
        self.assertIsNotNone(diff)
        self.assertGreater(diff, 35.0, f"Regional difference {diff} should be ~45 BPM")
        self.assertFalse(report["checks"]["regional_bpm_agreement"]["passed"])
        self.assertEqual(report["verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
