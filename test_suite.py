"""
SignalGuard Rigorous Test Suite — Tests A through E

Validates physiological signal processing mathematics, sampling rate invariance,
competing peak diagnostics, rejection of unusable/noisy signals, and regional consistency.
"""

import unittest
import numpy as np
import pandas as pd

import signalguard_utils as utils


def make_synthetic_rgb(
    duration_sec,
    fs,
    pulse_bpm,
    pulse_amp=0.015,
    noise_std=0.002,
    base_rgb=(140.0, 115.0, 95.0),
    motion_bpm=None,
    motion_amp=0.0,
):
    """
    Generate synthetic skin RGB traces containing known physiological and motion components.
    Physiological pulse manifests primarily in Green (absorption) with opposite phase in Red/Blue.
    """
    n = int(round(duration_sec * fs))
    t = np.arange(n) / fs

    r0, g0, b0 = base_rgb

    # Physiological pulse: PPG pulse is strongest in green channel (optical absorption of hemoglobin)
    f_pulse = pulse_bpm / 60.0
    pulse_wave = np.sin(2.0 * np.pi * f_pulse * t)

    # In rPPG: Green has negative PPG variation, while Red/Blue have smaller variation
    g_sig = g0 * (1.0 + pulse_amp * pulse_wave)
    r_sig = r0 * (1.0 - 0.3 * pulse_amp * pulse_wave)
    b_sig = b0 * (1.0 - 0.2 * pulse_amp * pulse_wave)

    # Optional motion artifact (modulates all channels in phase, especially luminance)
    if motion_bpm is not None and motion_amp > 0:
        f_motion = motion_bpm / 60.0
        motion_wave = np.sin(2.0 * np.pi * f_motion * t)
        r_sig += r0 * motion_amp * motion_wave
        g_sig += g0 * motion_amp * motion_wave
        b_sig += b0 * motion_amp * motion_wave

    # Noise
    if noise_std > 0:
        r_sig += np.random.normal(0, noise_std * r0, n)
        g_sig += np.random.normal(0, noise_std * g0, n)
        b_sig += np.random.normal(0, noise_std * b0, n)

    rgb = np.column_stack([r_sig, g_sig, b_sig])
    return t, rgb


class TestA_KnownFrequencySynthetic(unittest.TestCase):
    """Test A: Verify frequency and BPM recovery across the supported physiological band."""

    def test_known_frequencies(self):
        fs = 30.0
        duration = 20.0
        test_bpms = [50.0, 65.0, 75.0, 90.0, 110.0, 135.0, 160.0]
        tolerance_bpm = 1.8  # Well within physiological acceptance

        for target_bpm in test_bpms:
            np.random.seed(42)
            t, rgb = make_synthetic_rgb(duration, fs, target_bpm, pulse_amp=0.02, noise_std=0.001)
            raw_pos = utils.pos_from_rgb(rgb, fs)
            filt_pos = utils.filter_pos_signal(raw_pos, fs)
            metrics = utils.analyse_signal(filt_pos, fs)

            self.assertIsNotNone(metrics["bpm"], f"Failed to detect BPM for target {target_bpm}")
            err = abs(metrics["bpm"] - target_bpm)
            self.assertLess(
                err,
                tolerance_bpm,
                f"Target {target_bpm} BPM estimated as {metrics['bpm']:.2f} BPM (error {err:.2f} > {tolerance_bpm})",
            )
            self.assertTrue(
                utils.peak_is_clear(metrics),
                f"Peak was not clear for known clean signal at {target_bpm} BPM (peak_ratio={metrics['peak_ratio']})",
            )


class TestB_SamplingRateConsistency(unittest.TestCase):
    """Test B: Verify that estimated BPM is invariant to the sampling rate."""

    def test_sampling_rate_invariance(self):
        target_bpm = 76.0
        duration = 25.0
        sampling_rates = [15.0, 16.64, 20.0, 25.0, 30.0, 60.0]
        results = []

        for fs in sampling_rates:
            np.random.seed(123)
            t, rgb = make_synthetic_rgb(duration, fs, target_bpm, pulse_amp=0.02, noise_std=0.001)

            # Build dataframe to test timestamp inference
            df = pd.DataFrame({"time_seconds": t})
            inferred_fs = utils.infer_sampling_rate(df)
            self.assertAlmostEqual(inferred_fs, fs, delta=0.1)

            raw_pos = utils.pos_from_rgb(rgb, inferred_fs)
            filt_pos = utils.filter_pos_signal(raw_pos, inferred_fs)
            metrics = utils.analyse_signal(filt_pos, inferred_fs)

            self.assertIsNotNone(metrics["bpm"])
            results.append((fs, metrics["bpm"]))

        # Check maximum spread across all sampling rates
        bpms = [r[1] for r in results]
        spread = max(bpms) - min(bpms)
        self.assertLess(
            spread,
            1.5,
            f"Sampling rate variation caused excessive BPM spread: {spread:.2f} BPM across {results}",
        )


class TestC_CompetingSpectralPeaks(unittest.TestCase):
    """Test C: Verify handling of competing spectral peaks and motion de-weighting."""

    def test_competing_peaks_reported(self):
        fs = 30.0
        duration = 25.0
        pulse_bpm = 72.0
        motion_bpm = 130.0

        np.random.seed(999)
        # Signal with both pulse and strong motion
        t, rgb = make_synthetic_rgb(
            duration,
            fs,
            pulse_bpm=pulse_bpm,
            pulse_amp=0.015,
            motion_bpm=motion_bpm,
            motion_amp=0.025,
            noise_std=0.001,
        )

        # Tracking motion coordinate with dominant peak at 130 BPM
        tracking_motion = np.sin(2.0 * np.pi * (motion_bpm / 60.0) * t)

        raw_pos = utils.pos_from_rgb(rgb, fs)
        filt_pos = utils.filter_pos_signal(raw_pos, fs)

        # 1. Without motion signals: both candidates must be identified in candidate list
        metrics_no_m = utils.analyse_signal(filt_pos, fs)
        cand_bpms = [c["bpm"] for c in metrics_no_m["candidate_peaks"]]
        self.assertTrue(metrics_no_m["has_competing_peak"])
        # Verify both 72 and 130 exist in candidates
        has_pulse = any(abs(b - pulse_bpm) < 2.5 for b in cand_bpms)
        has_motion = any(abs(b - motion_bpm) < 2.5 for b in cand_bpms)
        self.assertTrue(has_pulse, f"Pulse candidate ~72 BPM missing from {cand_bpms}")
        self.assertTrue(has_motion, f"Motion candidate ~130 BPM missing from {cand_bpms}")

        # 2. With motion signals: motion frequency should be flagged and physiological candidate selected
        metrics_with_m = utils.analyse_signal(filt_pos, fs, motion_signals=tracking_motion)
        err = abs(metrics_with_m["bpm"] - pulse_bpm)
        self.assertLess(
            err,
            2.0,
            f"Expected physiological selection ~{pulse_bpm} BPM, got {metrics_with_m['bpm']} BPM",
        )
        # Verify motion candidate is flagged as motion
        motion_cands = [c for c in metrics_with_m["candidate_peaks"] if abs(c["bpm"] - motion_bpm) < 3.0]
        self.assertTrue(len(motion_cands) > 0 and motion_cands[0]["is_motion"])


class TestD_NoisyAndUnusableSignals(unittest.TestCase):
    """Test D: Verify pipeline rejects flat, pure noise, missing, or short signals."""

    def test_constant_signal(self):
        fs = 30.0
        t = np.arange(300) / fs
        # Constant RGB
        rgb = np.full((len(t), 3), 120.0)
        raw_pos = utils.pos_from_rgb(rgb, fs)
        filt_pos = utils.filter_pos_signal(raw_pos, fs)
        metrics = utils.analyse_signal(filt_pos, fs)
        self.assertIsNone(metrics["bpm"], "Constant signal should not yield a BPM")

    def test_random_noise(self):
        fs = 30.0
        np.random.seed(42)
        rgb = np.random.normal(120.0, 10.0, (int(20 * fs), 3))
        raw_pos = utils.pos_from_rgb(rgb, fs)
        filt_pos = utils.filter_pos_signal(raw_pos, fs)
        metrics = utils.analyse_signal(filt_pos, fs)
        # Random noise should either yield None or fail peak_is_clear
        if metrics["bpm"] is not None:
            self.assertFalse(
                utils.peak_is_clear(metrics),
                f"Random noise incorrectly passed peak_is_clear (ratio={metrics['peak_ratio']})",
            )

    def test_excessive_missing_samples(self):
        fs = 30.0
        t, rgb = make_synthetic_rgb(20.0, fs, 75.0)
        # Wipe out 70% of frames
        rgb[50:450, :] = np.nan
        raw_pos = utils.pos_from_rgb(rgb, fs)
        filt_pos = utils.filter_pos_signal(raw_pos, fs)
        metrics = utils.analyse_signal(filt_pos, fs)
        # With continuous valid segment < 8 seconds (only 50 samples = 1.6s remaining), metrics should be rejected
        self.assertIsNone(metrics["bpm"], "Signal with fragmented short segments should be rejected")

    def test_short_recording(self):
        fs = 30.0
        # 4 seconds duration (less than MIN_HR_SEGMENT_SECONDS = 8.0)
        t, rgb = make_synthetic_rgb(4.0, fs, 75.0)
        raw_pos = utils.pos_from_rgb(rgb, fs)
        filt_pos = utils.filter_pos_signal(raw_pos, fs)
        metrics = utils.analyse_signal(filt_pos, fs)
        self.assertIsNone(metrics["bpm"], "Short recording (< 8s) must be rejected")


class TestE_RegionalConsistency(unittest.TestCase):
    """Test E: Verify cross-regional consistency handling under noise differences and genuine disagreements."""

    def test_same_frequency_different_amplitudes_and_noise(self):
        fs = 30.0
        duration = 20.0
        pulse_bpm = 74.0

        np.random.seed(111)
        # Region A (forehead): clean, standard amplitude
        t, rgb_a = make_synthetic_rgb(duration, fs, pulse_bpm, pulse_amp=0.020, noise_std=0.001)
        # Region B (cheeks): lower amplitude, 3x higher noise
        t, rgb_b = make_synthetic_rgb(duration, fs, pulse_bpm, pulse_amp=0.010, noise_std=0.003)

        pos_a = utils.filter_pos_signal(utils.pos_from_rgb(rgb_a, fs), fs)
        pos_b = utils.filter_pos_signal(utils.pos_from_rgb(rgb_b, fs), fs)

        m_a = utils.analyse_signal(pos_a, fs)
        m_b = utils.analyse_signal(pos_b, fs)

        self.assertIsNotNone(m_a["bpm"])
        self.assertIsNotNone(m_b["bpm"])
        diff = abs(m_a["bpm"] - m_b["bpm"])
        self.assertLess(
            diff,
            3.0,
            f"Regions with same pulse should agree despite amplitude/noise difference (diff={diff:.2f})",
        )

    def test_genuinely_different_frequencies_not_forced(self):
        fs = 30.0
        duration = 20.0

        np.random.seed(222)
        # Region A at 70 BPM
        t, rgb_a = make_synthetic_rgb(duration, fs, 70.0, pulse_amp=0.02, noise_std=0.001)
        # Region B at 110 BPM
        t, rgb_b = make_synthetic_rgb(duration, fs, 110.0, pulse_amp=0.02, noise_std=0.001)

        pos_a = utils.filter_pos_signal(utils.pos_from_rgb(rgb_a, fs), fs)
        pos_b = utils.filter_pos_signal(utils.pos_from_rgb(rgb_b, fs), fs)

        m_a = utils.analyse_signal(pos_a, fs)
        m_b = utils.analyse_signal(pos_b, fs)

        diff = abs(m_a["bpm"] - m_b["bpm"])
        self.assertGreater(
            diff,
            30.0,
            f"Pipeline should NOT force agreement when regions have genuinely different frequencies (diff={diff:.2f})",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
