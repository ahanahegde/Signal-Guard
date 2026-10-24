"""
SignalGuard Centralized Digital Signal Processing (DSP) Foundation.

Authoritative implementation for:
- Timestamp sampling frequency inference and jitter diagnostics
- Explicit missing-data tracking, time-bounded gap interpolation, and gap statistics
- Wang et al. (2017) Plane-Orthogonal-to-Skin (POS) rPPG projection
- Zero-phase Butterworth bandpass filtering
- Multi-candidate spectral peak analysis with sub-bin parabolic interpolation,
  prominence, and local spectral signal-to-noise ratio (SNR)
- Motion-aware spectral analysis with overlap risk assessment and uncertainty tracking
- Multi-segment temporal evaluation and regional consistency metrics
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

# ============================================================
# CONFIGURABLE DSP DEFAULTS & THRESHOLDS (Provisional)
# ============================================================
LOW_HZ = 0.7  # 42 BPM
HIGH_HZ = 3.0  # 180 BPM

POS_WINDOW_SECONDS = 1.6
MIN_HR_SEGMENT_SECONDS = 8.0
MIN_CONTINUOUS_SECONDS = 8.0
MIN_FILTER_SEGMENT_SECONDS = 3.0

MIN_PEAK_RATIO = 2.0
MIN_MULTI_PEAK_RATIO = 1.20
MIN_LOCAL_SNR = 2.0
MAX_INTERP_GAP_SECONDS = 2.0
MAX_INTERPOLATED_FRACTION = 0.15
MOTION_MATCH_TOLERANCE_HZ = 0.05


def infer_sampling_rate(
    df: pd.DataFrame,
    return_details: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """
    Infer frames per second from actual video timestamps rather than nominal FPS metadata.
    Detects irregular sampling, frame jitter, and missing timestamps.

    Parameters:
        df: DataFrame containing 'time_seconds' column.
        return_details: If True, returns (fs, details_dict). If False, returns fs (float).
    """
    if "time_seconds" not in df.columns:
        raise ValueError("time_seconds column is missing. Re-run signal_extractor.py.")

    times = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(dtype=float)

    differences = np.diff(times)
    differences = differences[np.isfinite(differences) & (differences > 0)]

    if len(differences) == 0:
        raise ValueError("Cannot infer sampling rate: no positive timestamp differences found.")

    median_diff = float(np.median(differences))
    if median_diff <= 0:
        raise ValueError("Cannot infer sampling rate: median frame interval is non-positive.")

    fs = 1.0 / median_diff

    if fs <= 2 * HIGH_HZ:
        raise ValueError(
            f"Sampling rate {fs:.3f} Hz is below Nyquist limit for configured {HIGH_HZ:.1f} Hz cutoff."
        )

    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences))
    jitter_pct = float((std_diff / max(mean_diff, 1e-6)) * 100.0)

    details = {
        "sampling_rate_hz": fs,
        "median_dt_seconds": median_diff,
        "mean_dt_seconds": mean_diff,
        "std_dt_seconds": std_diff,
        "timing_jitter_pct": jitter_pct,
        "min_dt_seconds": float(np.min(differences)),
        "max_dt_seconds": float(np.max(differences)),
        "total_duration_seconds": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "total_frames": len(times),
    }

    if return_details:
        return fs, details
    return fs


def contiguous_runs(mask: Sequence[bool] | np.ndarray) -> list[np.ndarray]:
    """Find uninterrupted index runs where mask is True."""
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(indices) == 0:
        return []
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    return [run for run in np.split(indices, split_points) if len(run) > 0]


def compute_gap_statistics(
    data: np.ndarray | Sequence[float],
    fs: float,
    max_gap_seconds: float = MAX_INTERP_GAP_SECONDS,
    min_segment_seconds: float = MIN_HR_SEGMENT_SECONDS,
) -> dict[str, Any]:
    """
    Quantify valid samples, missing gaps, interpolatable gaps, and continuous segment lengths.
    Never assumes fixed sample count = fixed duration; computes strictly via time axis / fs.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        valid_mask = np.all(np.isfinite(arr), axis=1) & np.all(arr > 1e-6, axis=1)
    else:
        valid_mask = np.isfinite(arr)

    n_total = len(valid_mask)
    if n_total == 0:
        return {
            "total_samples": 0,
            "total_duration_sec": 0.0,
            "valid_samples": 0,
            "valid_duration_sec": 0.0,
            "valid_fraction": 0.0,
            "missing_samples": 0,
            "missing_duration_sec": 0.0,
            "missing_fraction": 1.0,
            "interpolated_samples_potential": 0,
            "interpolated_duration_potential_sec": 0.0,
            "interpolated_fraction_potential": 0.0,
            "max_gap_seconds": 0.0,
            "usable_segments_count": 0,
            "longest_continuous_seconds": 0.0,
        }

    n_valid = int(np.sum(valid_mask))
    n_missing = n_total - n_valid
    total_sec = float(n_total / fs)
    valid_sec = float(n_valid / fs)
    missing_sec = float(n_missing / fs)

    max_gap_samples = int(round(max_gap_seconds * fs))
    min_segment_samples = int(round(min_segment_seconds * fs))

    # Analyze invalid runs (gaps)
    invalid_runs = contiguous_runs(~valid_mask)
    first_valid = int(np.flatnonzero(valid_mask)[0]) if n_valid > 0 else -1
    last_valid = int(np.flatnonzero(valid_mask)[-1]) if n_valid > 0 else -1

    potential_interp_samples = 0
    max_gap_samples_found = 0

    for run in invalid_runs:
        run_len = len(run)
        if run_len > max_gap_samples_found:
            max_gap_samples_found = run_len
        # Only internal gaps within threshold are interpolatable
        if first_valid != -1 and run[0] > first_valid and run[-1] < last_valid:
            if run_len <= max_gap_samples:
                potential_interp_samples += run_len

    # Identify continuous segments after potential interpolation
    # A gap is bridgeable only if internal and <= max_gap_samples
    bridgeable_mask = valid_mask.copy()
    for run in invalid_runs:
        if first_valid != -1 and run[0] > first_valid and run[-1] < last_valid:
            if len(run) <= max_gap_samples:
                bridgeable_mask[run] = True

    post_interp_runs = contiguous_runs(bridgeable_mask)
    usable_runs = [r for r in post_interp_runs if len(r) >= min_segment_samples]
    longest_samples = max((len(r) for r in post_interp_runs), default=0)

    return {
        "total_samples": n_total,
        "total_duration_sec": total_sec,
        "valid_samples": n_valid,
        "valid_duration_sec": valid_sec,
        "valid_fraction": float(n_valid / n_total),
        "missing_samples": n_missing,
        "missing_duration_sec": missing_sec,
        "missing_fraction": float(n_missing / n_total),
        "interpolated_samples_potential": potential_interp_samples,
        "interpolated_duration_potential_sec": float(potential_interp_samples / fs),
        "interpolated_fraction_potential": float(potential_interp_samples / n_total),
        "max_gap_seconds": float(max_gap_samples_found / fs),
        "usable_segments_count": len(usable_runs),
        "longest_continuous_seconds": float(longest_samples / fs),
    }


def interpolate_invalid_samples(
    data: np.ndarray | Sequence[float],
    fs: float | None = None,
    max_gap_seconds: float = MAX_INTERP_GAP_SECONDS,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate short gaps of missing/NaN values within a time series.
    Enforces strict scientific safeguards:
    1. Gaps longer than max_gap_seconds remain NaN.
    2. Boundary gaps (at recording start or end) are NEVER extrapolated and remain NaN.
    3. Explicitly distinguishes original observations from interpolated samples via return_mask.

    Parameters:
        data: 1D or 2D array of signal values.
        fs: Sampling frequency in Hz (required for time-bounded gap limit).
        max_gap_seconds: Maximum allowed gap duration for interpolation in seconds.
        return_mask: If True, returns (interpolated_data, raw_valid_mask, interpolated_mask).
    """
    arr = np.asarray(data, dtype=float).copy()
    if arr.ndim == 1:
        out, raw_valid, interp_mask = _interpolate_1d_with_mask(arr, fs, max_gap_seconds)
        if return_mask:
            return out, raw_valid, interp_mask
        return out
    elif arr.ndim == 2:
        out = arr.copy()
        raw_valid = np.all(np.isfinite(arr), axis=1) & np.all(arr > 1e-6, axis=1)
        interp_mask = np.zeros(arr.shape[0], dtype=bool)

        for col in range(arr.shape[1]):
            col_out, _, col_interp = _interpolate_1d_with_mask(arr[:, col], fs, max_gap_seconds)
            out[:, col] = col_out
            interp_mask |= col_interp

        if return_mask:
            return out, raw_valid, interp_mask
        return out
    else:
        raise ValueError("Only 1D or 2D arrays are supported.")


def _interpolate_1d_with_mask(
    y: np.ndarray,
    fs: float | None = None,
    max_gap_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    raw_valid = np.isfinite(y)
    interp_mask = np.zeros(n, dtype=bool)

    if raw_valid.sum() < 4 or raw_valid.all():
        return y.copy(), raw_valid, interp_mask

    x = np.arange(n)
    interp_y = y.copy()

    # If no gap limit or fs not given, interpolate all internal NaNs
    first_valid = int(np.flatnonzero(raw_valid)[0])
    last_valid = int(np.flatnonzero(raw_valid)[-1])

    if max_gap_seconds is None or fs is None:
        internal_mask = ~raw_valid & (x >= first_valid) & (x <= last_valid)
        interp_y[internal_mask] = np.interp(x[internal_mask], x[raw_valid], y[raw_valid])
        interp_mask[internal_mask] = True
        return interp_y, raw_valid, interp_mask

    max_gap_samples = int(round(max_gap_seconds * fs))
    invalid_runs = contiguous_runs(~raw_valid)

    for run in invalid_runs:
        start = run[0]
        end = run[-1]
        # Only interpolate internal gaps within length limit; edge gaps remain NaN
        if start > first_valid and end < last_valid:
            if len(run) <= max_gap_samples:
                interp_y[run] = np.interp(run, x[raw_valid], y[raw_valid])
                interp_mask[run] = True

    return interp_y, raw_valid, interp_mask


def detect_rgb_artifacts(
    rgb: np.ndarray | Sequence[Sequence[float]],
    threshold: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect abrupt shifts in average RGB brightness (e.g. auto-exposure jumps, lighting flicker).
    Returns (artifact_mask, relative_change_series).
    """
    rgb_arr = np.asarray(rgb, dtype=float)
    n = len(rgb_arr)
    valid = np.all(np.isfinite(rgb_arr), axis=1)

    transitions = np.zeros(n, dtype=bool)
    relative_change = np.full(n, np.nan)

    for i in range(1, n):
        if not (valid[i - 1] and valid[i]):
            continue

        previous = float(np.mean(rgb_arr[i - 1]))
        current = float(np.mean(rgb_arr[i]))

        change = abs(current - previous) / max(abs(previous), 1.0)
        relative_change[i] = change

        if change > threshold:
            transitions[i] = True

    # Mark the transition and neighbouring boundary frames
    artifact_mask = transitions.copy()
    for i in np.flatnonzero(transitions):
        artifact_mask[max(0, i - 1) : min(n, i + 2)] = True

    # Never mark already-invalid samples as newly detected artifacts
    artifact_mask &= valid
    return artifact_mask, relative_change


def detect_motion_frequencies(
    motion_signals: np.ndarray | Sequence[float] | None,
    fs: float,
) -> list[float]:
    """
    Detect dominant frequencies in facial tracking motion signals (face center, width, pose).
    Returns sorted list of distinct motion frequency peaks in Hz.
    """
    if motion_signals is None:
        return []

    signals = np.asarray(motion_signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]

    motion_peaks: list[float] = []
    min_samples = int(round(MIN_FILTER_SEGMENT_SECONDS * fs))

    for col in range(signals.shape[1]):
        sig = signals[:, col]
        valid = np.isfinite(sig)
        if valid.sum() < min_samples:
            continue

        # Interpolate internal gaps for continuous motion spectrum estimation
        clean_sig, _, _ = _interpolate_1d_with_mask(sig, fs=fs, max_gap_seconds=MAX_INTERP_GAP_SECONDS)
        val_clean = np.isfinite(clean_sig)
        if val_clean.sum() < min_samples:
            continue

        clean_sig = clean_sig[val_clean]
        clean_sig = clean_sig - np.mean(clean_sig)
        if np.std(clean_sig) < 1e-8:
            continue

        nperseg = min(len(clean_sig), max(32, int(round(fs * 16.0))))
        nfft = max(4096, 2 ** int(np.ceil(np.log2(nperseg))))

        freqs, power = welch(
            clean_sig,
            fs=fs,
            nperseg=nperseg,
            nfft=nfft,
            detrend="linear",
            scaling="spectrum",
        )

        band_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        bf = freqs[band_mask]
        bp = power[band_mask]

        if len(bp) < 3:
            continue

        min_sep_bins = max(1, int(round(0.12 / (freqs[1] - freqs[0]))))
        peaks, _ = find_peaks(bp, distance=min_sep_bins)
        if len(peaks) > 0:
            top_idx = int(np.argmax(bp))
            mean_pwr = float(np.mean(bp))
            ratio = float(bp[top_idx]) / max(mean_pwr, 1e-12)
            # Flag significant motion peaks with prominence above background
            if ratio >= 3.0 and bp[top_idx] >= 1e-8:
                motion_peaks.append(float(bf[top_idx]))

    # Deduplicate peaks within tolerance
    unique_motion: list[float] = []
    for mf in motion_peaks:
        if not any(abs(mf - u) < MOTION_MATCH_TOLERANCE_HZ for u in unique_motion):
            unique_motion.append(mf)

    return sorted(unique_motion)


def pos_from_rgb(
    rgb: np.ndarray | Sequence[Sequence[float]],
    fs: float,
    window_seconds: float = POS_WINDOW_SECONDS,
    max_gap_seconds: float = MAX_INTERP_GAP_SECONDS,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract rPPG pulse signal using the Plane-Orthogonal-to-Skin (POS) algorithm (Wang et al., 2017).

    Equations:
      S1 = G - B
      S2 = -2R + G + B
      alpha = std(S1) / std(S2)
      P = S1 + alpha * S2

    Continuous overlapping windows (Hanning-weighted) are combined with overlap-add.
    Preserves explicit masks of valid and interpolated samples.

    Parameters:
        rgb: Nx3 array of R, G, B channel means.
        fs: Sampling rate in Hz.
        window_seconds: Duration of POS sliding projection window (default 1.6s).
        max_gap_seconds: Threshold for internal gap interpolation (default 2.0s).
        return_mask: If True, returns (pos_signal, raw_valid_mask, interpolated_mask).
    """
    rgb_arr = np.asarray(rgb, dtype=float)
    n = len(rgb_arr)
    output = np.full(n, np.nan)

    # Clean short gaps and extract explicit tracking masks
    clean_rgb, raw_valid, interp_mask = interpolate_invalid_samples(
        rgb_arr, fs=fs, max_gap_seconds=max_gap_seconds, return_mask=True
    )

    valid_rgb = np.all(np.isfinite(clean_rgb), axis=1) & np.all(clean_rgb > 1e-6, axis=1)

    window_size = max(16, int(round(window_seconds * fs)))
    hop = max(1, window_size // 2)

    for run in contiguous_runs(valid_rgb):
        segment = clean_rgb[run]
        if len(segment) < window_size:
            continue

        accum = np.zeros(len(segment))
        weight_sum = np.zeros(len(segment))

        starts = list(range(0, len(segment) - window_size + 1, hop))
        final_start = len(segment) - window_size
        if not starts or starts[-1] != final_start:
            starts.append(final_start)

        weights = np.maximum(np.hanning(window_size), 0.05)

        for start in starts:
            end = start + window_size
            block = segment[start:end]

            means = np.mean(block, axis=0)
            if np.any(means < 1e-6):
                continue

            normalized = block / means - 1.0
            r, g, b = normalized.T

            s1 = g - b
            s2 = -2.0 * r + g + b

            std_s2 = float(np.std(s2))
            std_s1 = float(np.std(s1))
            if std_s2 < 1e-8 or std_s1 < 1e-8:
                continue

            alpha = std_s1 / std_s2
            pulse = s1 + alpha * s2
            pulse -= np.mean(pulse)

            accum[start:end] += pulse * weights
            weight_sum[start:end] += weights

        combined = np.full(len(segment), np.nan)
        good = weight_sum > 0
        combined[good] = accum[good] / weight_sum[good]

        # Interpolate tiny boundary margins within segment if any
        val_comb = np.isfinite(combined)
        if val_comb.sum() >= 4 and not val_comb.all():
            x_seg = np.arange(len(segment))
            combined[~val_comb] = np.interp(x_seg[~val_comb], x_seg[val_comb], combined[val_comb])

        output[run] = combined

    if return_mask:
        return output, raw_valid, interp_mask
    return output


def filter_pos_signal(
    signal: np.ndarray | Sequence[float],
    fs: float,
    low: float = LOW_HZ,
    high: float = HIGH_HZ,
    max_gap_seconds: float = MAX_INTERP_GAP_SECONDS,
) -> np.ndarray:
    """
    Zero-phase bandpass-filter continuous POS segments using a 3rd-order Butterworth SOS filter.
    Interpolates internal gaps <= max_gap_seconds to avoid filter ringing and fragmentation.
    """
    signal_arr = np.asarray(signal, dtype=float)
    output = np.full(len(signal_arr), np.nan)

    clean_signal = interpolate_invalid_samples(
        signal_arr, fs=fs, max_gap_seconds=max_gap_seconds, return_mask=False
    )

    nyquist = fs / 2.0
    high_cutoff = min(high, nyquist * 0.95)
    if low <= 0 or high_cutoff <= low:
        raise ValueError(
            f"Invalid bandpass cutoffs {low:.3f}-{high_cutoff:.3f} Hz at sampling rate fs={fs:.3f} Hz."
        )

    sos = butter(3, [low, high_cutoff], btype="bandpass", fs=fs, output="sos")
    min_samples = max(24, int(round(MIN_FILTER_SEGMENT_SECONDS * fs)))

    for run in contiguous_runs(np.isfinite(clean_signal)):
        segment = clean_signal[run]
        if len(segment) < min_samples:
            continue

        segment = segment - np.mean(segment)
        if np.std(segment) < 1e-8:
            continue

        try:
            output[run] = sosfiltfilt(sos, segment)
        except ValueError:
            continue

    return output


def spectral_metrics(
    segment: np.ndarray | Sequence[float],
    fs: float,
    motion_signals: np.ndarray | Sequence[float] | None = None,
    min_hr_seconds: float = MIN_HR_SEGMENT_SECONDS,
) -> dict[str, Any]:
    """
    Estimate candidate heart rate, spectral prominence, local SNR, competing-peak ratios,
    motion artifact correlation, and candidate peaks list.
    """
    segment_arr = np.asarray(segment, dtype=float)

    empty: dict[str, Any] = {
        "bpm": None,
        "frequency_hz": None,
        "peak_ratio": None,
        "multi_peak_ratio": None,
        "local_snr": None,
        "prominence": None,
        "has_competing_peak": False,
        "candidate_peaks": [],
        "half_rate_power_ratio": None,
        "is_motion_contaminated": False,
        "motion_overlap_risk": "NONE",
        "motion_frequencies_hz": [],
        "uncertain": True,
        "selection_reason": "INSUFFICIENT_DURATION",
    }

    min_required_samples = int(round(min_hr_seconds * fs))
    if len(segment_arr) < min_required_samples:
        empty["selection_reason"] = f"Segment length ({len(segment_arr)}) < minimum required ({min_required_samples})"
        return empty

    if not np.all(np.isfinite(segment_arr)):
        # Interpolate short internal gaps if present
        segment_arr, _, _ = _interpolate_1d_with_mask(
            segment_arr, fs=fs, max_gap_seconds=MAX_INTERP_GAP_SECONDS
        )
        if not np.all(np.isfinite(segment_arr)):
            empty["selection_reason"] = "Segment contains unbridgeable NaNs"
            return empty

    segment_arr = segment_arr - np.mean(segment_arr)
    std_val = float(np.std(segment_arr))
    if std_val < 1e-8:
        empty["selection_reason"] = "Signal variance near zero"
        return empty

    duration = len(segment_arr) / fs
    nperseg = min(len(segment_arr), max(32, int(round(fs * min(duration, 16.0)))))
    nfft = max(4096, 2 ** int(np.ceil(np.log2(nperseg))))

    frequencies, power = welch(
        segment_arr,
        fs=fs,
        nperseg=nperseg,
        nfft=nfft,
        detrend="linear",
        scaling="spectrum",
    )

    band_mask = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)
    band_f = frequencies[band_mask]
    band_p = power[band_mask]

    if len(band_p) < 3:
        empty["selection_reason"] = "Physiological band contains insufficient frequency bins"
        return empty

    mean_power = float(np.mean(band_p))
    if mean_power <= 1e-12:
        empty["selection_reason"] = "Spectral band power near zero"
        return empty

    # Detect motion peaks from tracking coordinates if provided
    motion_f_hz = detect_motion_frequencies(motion_signals, fs)

    # Find distinct candidate spectral peaks
    min_separation_hz = max(0.12, 1.0 / duration)
    bin_width = float(frequencies[1] - frequencies[0])
    min_distance_bins = max(1, int(np.ceil(min_separation_hz / bin_width)))

    local_peaks, _ = find_peaks(band_p, distance=min_distance_bins)
    candidate_indices = set(int(i) for i in local_peaks)
    candidate_indices.add(int(np.argmax(band_p)))

    step_hz = float(band_f[1] - band_f[0])
    all_candidates: list[dict[str, Any]] = []

    for idx in candidate_indices:
        p_freq = float(band_f[idx])
        p_power = float(band_p[idx])

        # Parabolic interpolation for sub-bin frequency accuracy
        if 0 < idx < len(band_p) - 1:
            tiny = 1e-20
            left, centre, right = np.log(np.maximum(band_p[idx - 1 : idx + 2], tiny))
            denom = left - 2.0 * centre + right
            if abs(denom) > 1e-12:
                offset = 0.5 * (left - right) / denom
                offset = float(np.clip(offset, -1.0, 1.0))
                p_freq += offset * step_hz

        # Check proximity to known head motion frequencies
        is_motion = any(abs(p_freq - mf) < MOTION_MATCH_TOLERANCE_HZ for mf in motion_f_hz)

        # Compute local spectral SNR around this peak
        local_mask = (band_f >= p_freq - 0.35) & (band_f <= p_freq + 0.35)
        lobe_mask = (band_f >= p_freq - 0.08) & (band_f <= p_freq + 0.08)
        surround_mask = local_mask & (~lobe_mask)
        if np.any(surround_mask):
            local_noise = float(np.median(band_p[surround_mask]))
        else:
            local_noise = mean_power
        local_snr = p_power / max(local_noise, 1e-12)

        # Peak prominence calculation
        left_min = float(np.min(band_p[: idx + 1])) if idx > 0 else 0.0
        right_min = float(np.min(band_p[idx:])) if idx < len(band_p) - 1 else 0.0
        prominence = p_power - max(left_min, right_min)

        all_candidates.append({
            "idx": idx,
            "frequency_hz": p_freq,
            "bpm": float(p_freq * 60.0),
            "power": p_power,
            "local_snr": float(local_snr),
            "prominence": float(prominence),
            "is_motion": is_motion,
        })

    # Sort candidates by raw power descending
    all_candidates.sort(key=lambda c: c["power"], reverse=True)

    if not all_candidates:
        empty["selection_reason"] = "No distinct spectral peaks found"
        return empty

    top_cand = all_candidates[0]

    # Half-rate / 2nd-harmonic power diagnostic for top candidate
    half_freq = top_cand["frequency_hz"] / 2.0
    half_power = 0.0
    if LOW_HZ <= half_freq <= HIGH_HZ:
        half_power = float(np.interp(half_freq, band_f, band_p))
    half_rate_ratio = half_power / max(top_cand["power"], 1e-12)

    # Candidate selection:
    # If top peak is free from motion contamination, select it
    # If top peak is motion-contaminated, inspect non-motion alternatives
    selected = top_cand
    overlap_risk = "NONE"
    selection_reason = "DOMINANT_PEAK_CLEAN"
    uncertain = False

    if top_cand["is_motion"]:
        overlap_risk = "HIGH"
        non_motion = [c for c in all_candidates if not c["is_motion"]]
        if non_motion:
            best_non_motion = non_motion[0]
            # Require the physiological candidate to have substantial power relative to the dominant peak
            if (
                best_non_motion["power"] >= 0.25 * top_cand["power"]
                and (best_non_motion["power"] / mean_power) >= 1.8
                and best_non_motion["local_snr"] >= MIN_LOCAL_SNR
            ):
                selected = best_non_motion
                overlap_risk = "MODERATE"
                selection_reason = "MOTION_DE_WEIGHTED_PHYSIOLOGICAL_ALTERNATIVE_SELECTED"
            else:
                selected = top_cand
                overlap_risk = "HIGH"
                selection_reason = "DOMINANT_PEAK_SWAMPED_BY_MOTION_NO_VIABLE_ALTERNATIVE"
                uncertain = True
        else:
            selection_reason = "ALL_PEAKS_MOTION_CONTAMINATED"
            uncertain = True

    top_power = selected["power"]
    peak_ratio = top_power / mean_power

    # Competing peak analysis relative to selected candidate
    competitors = [
        c for c in all_candidates
        if abs(c["frequency_hz"] - selected["frequency_hz"]) >= min_separation_hz
    ]

    multi_peak_ratio = None
    has_competitor = False
    if competitors:
        has_competitor = True
        second_best = max(competitors, key=lambda c: c["power"])
        second_power = second_best["power"]
        if second_power > 1e-12:
            multi_peak_ratio = top_power / second_power
            # Flag ambiguity if competitor is nearly equal in power without distinguishable physiological evidence
            if multi_peak_ratio < 1.15:
                uncertain = True
                selection_reason = "AMBIGUOUS_COMPETING_PEAKS"

    candidate_peaks_summary = [
        {
            "bpm": round(c["bpm"], 1),
            "frequency_hz": round(c["frequency_hz"], 4),
            "power": c["power"],
            "local_snr": round(c["local_snr"], 2),
            "prominence": c["prominence"],
            "is_motion": c["is_motion"],
        }
        for c in all_candidates[:5]
    ]

    return {
        "bpm": float(selected["bpm"]),
        "frequency_hz": float(selected["frequency_hz"]),
        "peak_ratio": float(peak_ratio),
        "multi_peak_ratio": float(multi_peak_ratio) if multi_peak_ratio is not None else None,
        "local_snr": float(selected["local_snr"]),
        "prominence": float(selected["prominence"]),
        "has_competing_peak": has_competitor,
        "candidate_peaks": candidate_peaks_summary,
        "half_rate_power_ratio": float(half_rate_ratio),
        "is_motion_contaminated": bool(selected["is_motion"]),
        "motion_overlap_risk": overlap_risk,
        "motion_frequencies_hz": [round(f, 4) for f in motion_f_hz],
        "uncertain": bool(uncertain),
        "selection_reason": selection_reason,
    }


def analyse_signal(
    signal: np.ndarray | Sequence[float],
    fs: float,
    motion_signals: np.ndarray | Sequence[float] | None = None,
    max_gap_seconds: float = MAX_INTERP_GAP_SECONDS,
    min_hr_seconds: float = MIN_HR_SEGMENT_SECONDS,
) -> dict[str, Any]:
    """
    Measure signal coverage, compute gap statistics, evaluate all usable segments,
    and analyze physiological spectral metrics.
    """
    signal_arr = np.asarray(signal, dtype=float)
    gap_stats = compute_gap_statistics(
        signal_arr, fs, max_gap_seconds=max_gap_seconds, min_segment_seconds=min_hr_seconds
    )

    result: dict[str, Any] = {
        "coverage": float(gap_stats["valid_fraction"] * 100.0),
        "missing_fraction": gap_stats["missing_fraction"],
        "interpolated_fraction": gap_stats["interpolated_fraction_potential"],
        "longest_seconds": gap_stats["longest_continuous_seconds"],
        "segment_samples": 0,
        "usable_segments_count": gap_stats["usable_segments_count"],
        "usable_segments": [],
        "bpm": None,
        "frequency_hz": None,
        "peak_ratio": None,
        "multi_peak_ratio": None,
        "local_snr": None,
        "prominence": None,
        "has_competing_peak": False,
        "candidate_peaks": [],
        "half_rate_power_ratio": None,
        "is_motion_contaminated": False,
        "motion_overlap_risk": "NONE",
        "motion_frequencies_hz": [],
        "uncertain": True,
        "selection_reason": "NO_VALID_DATA",
        "gap_statistics": gap_stats,
    }

    if len(signal_arr) == 0 or gap_stats["valid_samples"] == 0:
        return result

    # Interpolate short internal gaps to preserve physiological continuity
    clean_signal = interpolate_invalid_samples(
        signal_arr, fs=fs, max_gap_seconds=max_gap_seconds, return_mask=False
    )
    runs = contiguous_runs(np.isfinite(clean_signal))

    min_samples = int(round(min_hr_seconds * fs))
    usable_runs = [r for r in runs if len(r) >= min_samples]

    if not usable_runs:
        result["selection_reason"] = (
            f"Longest continuous segment ({gap_stats['longest_continuous_seconds']:.2f}s) "
            f"< minimum required ({min_hr_seconds:.1f}s)"
        )
        return result

    # Evaluate each usable segment
    segment_evals: list[dict[str, Any]] = []
    ms = np.asarray(motion_signals, dtype=float) if motion_signals is not None else None

    for run_idx, run in enumerate(usable_runs):
        seg = clean_signal[run]
        seg_motion = ms[run] if ms is not None and len(ms) == len(clean_signal) else None
        seg_m = spectral_metrics(seg, fs, motion_signals=seg_motion, min_hr_seconds=min_hr_seconds)
        segment_evals.append({
            "segment_index": run_idx,
            "start_sample": int(run[0]),
            "end_sample": int(run[-1]),
            "duration_sec": float(len(run) / fs),
            "bpm": seg_m["bpm"],
            "peak_ratio": seg_m["peak_ratio"],
            "local_snr": seg_m.get("local_snr"),
            "uncertain": seg_m.get("uncertain", False),
        })

    result["usable_segments"] = segment_evals

    # Analyze primary (longest) continuous segment
    longest = max(usable_runs, key=len)
    segment = clean_signal[longest]

    result["longest_seconds"] = float(len(segment) / fs)
    result["segment_samples"] = len(segment)

    seg_motion = ms[longest] if ms is not None and len(ms) == len(clean_signal) else None
    metrics = spectral_metrics(segment, fs, motion_signals=seg_motion, min_hr_seconds=min_hr_seconds)
    result.update(metrics)

    return result


def peak_is_clear(metrics: dict[str, Any]) -> bool:
    """
    Evaluate whether a spectral peak is sufficiently clear and unambiguous.
    Requires:
      1. Peak exists and is not marked uncertain.
      2. Peak/mean ratio >= MIN_PEAK_RATIO.
      3. Dominant/competing-peak ratio >= MIN_MULTI_PEAK_RATIO.
    """
    if (
        metrics.get("bpm") is None
        or metrics.get("peak_ratio") is None
        or metrics["peak_ratio"] < MIN_PEAK_RATIO
        or metrics.get("uncertain", False)
    ):
        return False

    ratio = metrics.get("multi_peak_ratio")
    if ratio is None:
        return True

    return ratio >= MIN_MULTI_PEAK_RATIO
