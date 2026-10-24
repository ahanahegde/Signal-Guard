import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch, find_peaks


LOW_HZ = 0.7
HIGH_HZ = 3.0

POS_WINDOW_SECONDS = 1.6
MIN_HR_SEGMENT_SECONDS = 8.0
MIN_FILTER_SEGMENT_SECONDS = 3.0

MIN_PEAK_RATIO = 2.0
MIN_MULTI_PEAK_RATIO = 1.20
MAX_INTERP_GAP_SECONDS = 2.0
MOTION_MATCH_TOLERANCE_HZ = 0.05


def infer_sampling_rate(df):
    """Infer frames per second from actual video timestamps."""
    if "time_seconds" not in df.columns:
        raise ValueError(
            "time_seconds is missing. Re-run signal_extractor.py."
        )

    times = pd.to_numeric(
        df["time_seconds"], errors="coerce"
    ).to_numpy(dtype=float)

    differences = np.diff(times)
    differences = differences[
        np.isfinite(differences) & (differences > 0)
    ]

    if len(differences) == 0:
        raise ValueError("Cannot infer sampling rate from timestamps.")

    fs = 1.0 / float(np.median(differences))

    if fs <= 2 * HIGH_HZ:
        raise ValueError(
            f"Sampling rate {fs:.3f} Hz is too low for "
            f"the configured {HIGH_HZ:.1f} Hz upper frequency."
        )

    return fs


def contiguous_runs(mask):
    """Find uninterrupted runs of True values."""
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))

    if len(indices) == 0:
        return []

    split_points = np.where(np.diff(indices) > 1)[0] + 1

    return [
        run for run in np.split(indices, split_points)
        if len(run) > 0
    ]


def interpolate_invalid_samples(data, fs=None, max_gap_seconds=MAX_INTERP_GAP_SECONDS):
    """
    Linearly interpolate short gaps of missing/NaN values within a time series.
    Gaps longer than max_gap_seconds (if specified and fs provided) remain NaN.
    """
    arr = np.asarray(data, dtype=float).copy()
    if arr.ndim == 1:
        return _interpolate_1d(arr, fs, max_gap_seconds)
    elif arr.ndim == 2:
        out = arr.copy()
        for col in range(arr.shape[1]):
            out[:, col] = _interpolate_1d(arr[:, col], fs, max_gap_seconds)
        return out
    else:
        raise ValueError("Only 1D or 2D arrays are supported.")


def _interpolate_1d(y, fs=None, max_gap_seconds=None):
    n = len(y)
    valid = np.isfinite(y)
    if valid.sum() < 4 or valid.all():
        return y.copy()

    x = np.arange(n)
    interp_y = y.copy()
    
    # If no gap limit or fs not given, interpolate all internal NaNs
    if max_gap_seconds is None or fs is None:
        first_valid = np.flatnonzero(valid)[0]
        last_valid = np.flatnonzero(valid)[-1]
        internal_mask = ~valid & (x >= first_valid) & (x <= last_valid)
        interp_y[internal_mask] = np.interp(x[internal_mask], x[valid], y[valid])
        return interp_y

    max_gap_samples = int(round(max_gap_seconds * fs))
    # Identify invalid runs
    invalid_runs = contiguous_runs(~valid)
    first_valid = np.flatnonzero(valid)[0]
    last_valid = np.flatnonzero(valid)[-1]

    for run in invalid_runs:
        start = run[0]
        end = run[-1]
        # Only interpolate internal gaps within length limit
        if start > first_valid and end < last_valid:
            if len(run) <= max_gap_samples:
                interp_y[run] = np.interp(run, x[valid], y[valid])

    return interp_y


def detect_rgb_artifacts(rgb, threshold=0.10):
    """
    Heuristic detector for abrupt changes in average RGB brightness.
    """
    rgb = np.asarray(rgb, dtype=float)

    n = len(rgb)
    valid = np.all(np.isfinite(rgb), axis=1)

    transitions = np.zeros(n, dtype=bool)
    relative_change = np.full(n, np.nan)

    for i in range(1, n):
        if not (valid[i - 1] and valid[i]):
            continue

        previous = float(np.mean(rgb[i - 1]))
        current = float(np.mean(rgb[i]))

        change = abs(current - previous) / max(abs(previous), 1.0)
        relative_change[i] = change

        if change > threshold:
            transitions[i] = True

    # Mark the transition and immediately neighbouring frames.
    artifact_mask = transitions.copy()

    for i in np.flatnonzero(transitions):
        artifact_mask[max(0, i - 1):min(n, i + 2)] = True

    # Never mark already-invalid samples as newly detected artifacts.
    artifact_mask &= valid

    return artifact_mask, relative_change


def detect_motion_frequencies(motion_signals, fs):
    """
    Detect dominant frequencies in facial tracking motion signals.
    """
    if motion_signals is None:
        return []

    signals = np.asarray(motion_signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]

    motion_peaks = []
    min_samples = int(round(MIN_FILTER_SEGMENT_SECONDS * fs))

    for col in range(signals.shape[1]):
        sig = signals[:, col]
        valid = np.isfinite(sig)
        if valid.sum() < min_samples:
            continue
        
        # Interpolate internal gaps for continuous motion analysis
        clean_sig = _interpolate_1d(sig, fs=fs, max_gap_seconds=MAX_INTERP_GAP_SECONDS)
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
            # Gross motion peaks with high prominence above spectral background
            if ratio >= 3.5 and bp[top_idx] >= 1e-8:
                motion_peaks.append(float(bf[top_idx]))

    # Deduplicate peaks within tolerance
    unique_motion = []
    for mf in motion_peaks:
        if not any(abs(mf - u) < MOTION_MATCH_TOLERANCE_HZ for u in unique_motion):
            unique_motion.append(mf)

    return sorted(unique_motion)


def pos_from_rgb(rgb, fs, window_seconds=POS_WINDOW_SECONDS, max_gap_seconds=MAX_INTERP_GAP_SECONDS):
    """
    Extract a candidate POS signal from RGB samples using standard Wang et al. (2017)
    POS projection with Hanning window overlap-add.

    Short tracking gaps (<= max_gap_seconds) are interpolated per channel to preserve
    temporal continuity. Missing sections longer than max_gap_seconds remain NaN.
    """
    rgb = np.asarray(rgb, dtype=float)
    n = len(rgb)
    output = np.full(n, np.nan)

    # First interpolate short gaps per channel
    clean_rgb = interpolate_invalid_samples(rgb, fs=fs, max_gap_seconds=max_gap_seconds)

    valid_rgb = (
        np.all(np.isfinite(clean_rgb), axis=1) &
        np.all(clean_rgb > 1e-6, axis=1)
    )

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
            if std_s2 < 1e-8:
                continue

            alpha = float(np.std(s1)) / std_s2
            pulse = s1 + alpha * s2
            pulse -= np.mean(pulse)

            accum[start:end] += pulse * weights
            weight_sum[start:end] += weights

        combined = np.full(len(segment), np.nan)
        good = weight_sum > 0
        combined[good] = accum[good] / weight_sum[good]

        # Interpolate any tiny boundary edge within the segment
        val_comb = np.isfinite(combined)
        if val_comb.sum() >= 4 and not val_comb.all():
            x_seg = np.arange(len(segment))
            combined[~val_comb] = np.interp(x_seg[~val_comb], x_seg[val_comb], combined[val_comb])

        output[run] = combined

    return output


def filter_pos_signal(signal, fs, low=LOW_HZ, high=HIGH_HZ, max_gap_seconds=MAX_INTERP_GAP_SECONDS):
    """
    Bandpass-filter continuous POS segments independently.
    Interpolates short internal gaps to avoid filter ringing and fragmentation.
    """
    signal = np.asarray(signal, dtype=float)
    output = np.full(len(signal), np.nan)

    clean_signal = interpolate_invalid_samples(signal, fs=fs, max_gap_seconds=max_gap_seconds)

    nyquist = fs / 2.0
    high_cutoff = min(high, nyquist * 0.95)
    if low <= 0 or high_cutoff <= low:
        raise ValueError(f"Invalid band-pass {low:.3f}-{high_cutoff:.3f} Hz at fs={fs:.3f} Hz.")

    sos = butter(
        3,
        [low, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    min_samples = max(24, int(round(MIN_FILTER_SEGMENT_SECONDS * fs)))

    for run in contiguous_runs(np.isfinite(clean_signal)):
        segment = clean_signal[run]

        if len(segment) < min_samples:
            continue

        segment = segment - np.mean(segment)

        try:
            output[run] = sosfiltfilt(sos, segment)
        except ValueError:
            continue

    return output


def spectral_metrics(segment, fs, motion_signals=None, min_hr_seconds=MIN_HR_SEGMENT_SECONDS):
    """
    Estimate the candidate heart rate, spectral peak ratio, competing-peak ratio,
    motion artifact correlation, and candidate peaks list.
    """
    segment = np.asarray(segment, dtype=float)

    empty = {
        "bpm": None,
        "frequency_hz": None,
        "peak_ratio": None,
        "multi_peak_ratio": None,
        "has_competing_peak": False,
        "candidate_peaks": [],
        "half_rate_power_ratio": None,
        "is_motion_contaminated": False,
        "motion_frequencies_hz": [],
    }

    if len(segment) < int(round(min_hr_seconds * fs)):
        return empty

    if not np.all(np.isfinite(segment)):
        # Interpolate short internal gaps if present
        segment = _interpolate_1d(segment, fs=fs, max_gap_seconds=MAX_INTERP_GAP_SECONDS)
        if not np.all(np.isfinite(segment)):
            return empty

    segment = segment - np.mean(segment)

    if np.std(segment) < 1e-8:
        return empty

    duration = len(segment) / fs
    nperseg = min(
        len(segment),
        max(32, int(round(fs * min(duration, 16.0)))),
    )

    nfft = max(4096, 2 ** int(np.ceil(np.log2(nperseg))))

    frequencies, power = welch(
        segment,
        fs=fs,
        nperseg=nperseg,
        nfft=nfft,
        detrend="linear",
        scaling="spectrum",
    )

    band_mask = (
        (frequencies >= LOW_HZ) &
        (frequencies <= HIGH_HZ)
    )

    band_f = frequencies[band_mask]
    band_p = power[band_mask]

    if len(band_p) < 3:
        return empty

    mean_power = float(np.mean(band_p))
    if mean_power <= 1e-12:
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

    # Compute sub-bin interpolated frequency and power for each peak
    step_hz = float(band_f[1] - band_f[0])
    all_candidates = []

    for idx in candidate_indices:
        p_freq = float(band_f[idx])
        p_power = float(band_p[idx])

        # Parabolic interpolation for sub-bin frequency accuracy
        if 0 < idx < len(band_p) - 1:
            tiny = 1e-20
            left, centre, right = np.log(np.maximum(band_p[idx - 1:idx + 2], tiny))
            denom = left - 2.0 * centre + right
            if abs(denom) > 1e-12:
                offset = 0.5 * (left - right) / denom
                offset = float(np.clip(offset, -1.0, 1.0))
                p_freq += offset * step_hz

        # Check proximity to known head motion frequencies
        is_motion = any(abs(p_freq - mf) < MOTION_MATCH_TOLERANCE_HZ for mf in motion_f_hz)

        all_candidates.append({
            "idx": idx,
            "frequency_hz": p_freq,
            "bpm": float(p_freq * 60.0),
            "power": p_power,
            "is_motion": is_motion,
        })

    # Sort candidates by raw power descending
    all_candidates.sort(key=lambda c: c["power"], reverse=True)

    if not all_candidates:
        return empty

    # Half-rate / 2nd-harmonic power diagnostic for top candidate
    top_cand = all_candidates[0]
    half_freq = top_cand["frequency_hz"] / 2.0
    half_power = 0.0
    if LOW_HZ <= half_freq <= HIGH_HZ:
        half_power = float(np.interp(half_freq, band_f, band_p))
    half_rate_ratio = half_power / max(top_cand["power"], 1e-12)

    # Candidate selection:
    # If the strongest peak is not motion-contaminated, select it.
    # If the strongest peak IS motion-contaminated, inspect if a valid non-motion candidate
    # has substantial physiological power (>= 20% of top power or peak ratio >= 1.8).
    selected = top_cand
    if top_cand["is_motion"]:
        non_motion = [c for c in all_candidates if not c["is_motion"]]
        if non_motion:
            best_non_motion = non_motion[0]
            # Require the physiological candidate to have substantial power
            if best_non_motion["power"] >= 0.20 * top_cand["power"] or (best_non_motion["power"] / mean_power) >= 1.8:
                selected = best_non_motion

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

    candidate_peaks_summary = [
        {
            "bpm": round(c["bpm"], 1),
            "frequency_hz": round(c["frequency_hz"], 4),
            "power": c["power"],
            "is_motion": c["is_motion"],
        }
        for c in all_candidates[:5]
    ]

    return {
        "bpm": float(selected["bpm"]),
        "frequency_hz": float(selected["frequency_hz"]),
        "peak_ratio": float(peak_ratio),
        "multi_peak_ratio": float(multi_peak_ratio) if multi_peak_ratio is not None else None,
        "has_competing_peak": has_competitor,
        "candidate_peaks": candidate_peaks_summary,
        "half_rate_power_ratio": float(half_rate_ratio),
        "is_motion_contaminated": bool(selected["is_motion"]),
        "motion_frequencies_hz": [round(f, 4) for f in motion_f_hz],
    }


def analyse_signal(signal, fs, motion_signals=None, max_gap_seconds=MAX_INTERP_GAP_SECONDS, min_hr_seconds=MIN_HR_SEGMENT_SECONDS):
    """
    Measure coverage and analyse the longest continuous valid run.
    Interpolates short internal gaps up to max_gap_seconds.
    """
    signal = np.asarray(signal, dtype=float)
    valid_initial = np.isfinite(signal)
    coverage = float(valid_initial.mean() * 100) if len(signal) else 0.0

    result = {
        "coverage": coverage,
        "longest_seconds": 0.0,
        "segment_samples": 0,
        "bpm": None,
        "frequency_hz": None,
        "peak_ratio": None,
        "multi_peak_ratio": None,
        "has_competing_peak": False,
        "candidate_peaks": [],
        "half_rate_power_ratio": None,
        "is_motion_contaminated": False,
        "motion_frequencies_hz": [],
    }

    if len(signal) == 0 or not np.any(valid_initial):
        return result

    # Interpolate short internal gaps to preserve physiological continuity
    clean_signal = interpolate_invalid_samples(signal, fs=fs, max_gap_seconds=max_gap_seconds)
    runs = contiguous_runs(np.isfinite(clean_signal))

    if not runs:
        return result

    longest = max(runs, key=len)
    segment = clean_signal[longest]

    result["longest_seconds"] = len(segment) / fs
    result["segment_samples"] = len(segment)

    # Slice motion signals matching the longest run if provided
    seg_motion = None
    if motion_signals is not None:
        ms = np.asarray(motion_signals, dtype=float)
        if len(ms) == len(clean_signal):
            seg_motion = ms[longest]

    metrics = spectral_metrics(segment, fs, motion_signals=seg_motion, min_hr_seconds=min_hr_seconds)
    result.update(metrics)

    return result


def peak_is_clear(metrics):
    """
    Require an above-background peak. When no distinct competitor exists,
    the peak-to-mean ratio remains the available dominance measure.
    """
    if (
        metrics["bpm"] is None
        or metrics["peak_ratio"] is None
        or metrics["peak_ratio"] < MIN_PEAK_RATIO
    ):
        return False

    ratio = metrics["multi_peak_ratio"]

    if ratio is None:
        return True

    return ratio >= MIN_MULTI_PEAK_RATIO
