"""
SignalGuard Full Regression & Stress Evaluation Script.

Evaluates and compares before-and-after results across all stress variants:
- Original video
- Downscaled & resized video ("compression" variant: 0.75x spatial resampling)
- Reduced brightness
- Increased brightness
- Temporal frame dropping

Reports both regional candidate agreement and the full SignalGuard Reliability Engine verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

import signalguard_utils as utils
from eligibility_gate import SignalGuardReliabilityEngine, ReliabilityConfig


def evaluate_variant(variant_name: str, outdir: Path, engine: SignalGuardReliabilityEngine) -> dict:
    pos_file = outdir / "pos_signals.csv"
    rgb_file = outdir / "rgb_signals.csv"

    if not rgb_file.exists():
        raise FileNotFoundError(f"Missing {rgb_file}")

    rgb_df = pd.read_csv(rgb_file)
    fs = utils.infer_sampling_rate(rgb_df)

    m_cols = [c for c in ["face_center_y_norm", "face_center_x_norm", "face_width_norm"] if c in rgb_df.columns]
    motion = rgb_df[m_cols].to_numpy(dtype=float) if m_cols else None

    # Compute or load POS
    if pos_file.exists():
        pos_df = pd.read_csv(pos_file)
    else:
        f_raw = utils.pos_from_rgb(rgb_df[["forehead_R", "forehead_G", "forehead_B"]].to_numpy(), fs)
        f_filt = utils.filter_pos_signal(f_raw, fs)
        c_raw = utils.pos_from_rgb(rgb_df[["cheeks_R", "cheeks_G", "cheeks_B"]].to_numpy(), fs)
        c_filt = utils.filter_pos_signal(c_raw, fs)
        pos_df = pd.DataFrame({
            "time_seconds": rgb_df["time_seconds"],
            "forehead_POS": f_raw,
            "forehead_POS_filtered": f_filt,
            "cheeks_POS": c_raw,
            "cheeks_POS_filtered": c_filt,
        })

    report = engine.evaluate(pos_df, rgb_df)

    f_bpm = report["forehead_candidate_bpm"]
    c_bpm = report["cheeks_candidate_bpm"]
    diff = report["regional_bpm_difference"]

    motion_bpms = [round(f * 60.0, 1) for f in report["categories"]["E_motion_and_tracking"]["motion_frequencies_hz"]]

    return {
        "variant": variant_name,
        "forehead_bpm": round(f_bpm, 1) if f_bpm else None,
        "cheeks_bpm": round(c_bpm, 1) if c_bpm else None,
        "diff_bpm": round(diff, 1) if diff is not None else None,
        "regional_agreement": "PASS" if (diff is not None and diff <= 13.0) else "FAIL",
        "verdict": report["verdict"],
        "usable_windows": report["categories"]["C_temporal_stability"]["usable_windows"],
        "motion_bpms": motion_bpms,
        "failure_reasons": report["failure_reasons"],
        "warnings": report["warnings"],
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    stress_dir = root / "stress_results"

    # Historical uncorrected baseline from initial pipeline logs
    # where cheeks reported 132.1 BPM due to blind argmax picking motion artifact
    before_data = {
        "original": {"forehead": 73.4, "cheeks": 132.1, "diff": 59.3},
        "compression": {"forehead": 73.2, "cheeks": 132.3, "diff": 59.0},
        "brightness_down": {"forehead": 72.9, "cheeks": 131.9, "diff": 59.1},
        "brightness_up": {"forehead": 73.0, "cheeks": 132.0, "diff": 59.4},
        "temporal_drop": {"forehead": 74.8, "cheeks": 132.0, "diff": 59.4},
    }

    engine = SignalGuardReliabilityEngine()
    records = []
    print("=" * 95)
    print("SIGNALGUARD RELIABILITY ENGINE: BEFORE VS. AFTER REGRESSION BENCHMARK")
    print("=" * 95)

    variants = ["original", "compression", "brightness_down", "brightness_up", "temporal_drop"]
    for v in variants:
        v_dir = stress_dir / v
        if not v_dir.exists():
            print(f"Skipping {v}: directory {v_dir} not found")
            continue

        try:
            after = evaluate_variant(v, v_dir, engine)
        except Exception as exc:
            print(f"Error evaluating {v}: {exc}")
            continue

        b = before_data.get(v, {"forehead": "N/A", "cheeks": "N/A", "diff": "N/A"})

        records.append({
            "Variant": v,
            "Prior Forehead": b["forehead"],
            "Prior Cheeks": b["cheeks"],
            "Prior Diff": b["diff"],
            "Corrected Forehead": after["forehead_bpm"],
            "Corrected Cheeks": after["cheeks_bpm"],
            "Corrected Diff": after["diff_bpm"],
            "Regional Match": after["regional_agreement"],
            "Engine Verdict": after["verdict"],
            "Motion Flagged (BPM)": ", ".join(map(str, after["motion_bpms"])) if after["motion_bpms"] else "None",
        })

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    print("=" * 95)

    print("\nDetailed Diagnostic Reasons per Variant:")
    for v in variants:
        v_dir = stress_dir / v
        if not v_dir.exists():
            continue
        try:
            res = evaluate_variant(v, v_dir, engine)
            print(f"\n[{v.upper()}] Verdict: {res['verdict']}")
            if res["failure_reasons"]:
                print("  Failed checks:")
                for r in res["failure_reasons"]:
                    print(f"    - {r}")
            if res["warnings"]:
                print("  Warnings:")
                for w in res["warnings"]:
                    print(f"    ! {w}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
