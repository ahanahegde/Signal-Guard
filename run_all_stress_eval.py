"""
SignalGuard Full Regression Evaluation Script.

Evaluates and compares before-and-after results for:
- Original video
- Compression variant
- Brightness down variant
- Brightness up variant
- Temporal drop variant
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

import signalguard_utils as utils


def evaluate_variant(variant_name, csv_path):
    df = pd.read_csv(csv_path)
    fs = utils.infer_sampling_rate(df)
    
    m_cols = [c for c in ['face_center_y_norm', 'face_center_x_norm', 'face_width_norm'] if c in df.columns]
    motion = df[m_cols].to_numpy(dtype=float) if m_cols else None
    
    f_raw = utils.pos_from_rgb(df[['forehead_R', 'forehead_G', 'forehead_B']].to_numpy(), fs)
    f_filt = utils.filter_pos_signal(f_raw, fs)
    f_m = utils.analyse_signal(f_filt, fs, motion_signals=motion)
    
    c_raw = utils.pos_from_rgb(df[['cheeks_R', 'cheeks_G', 'cheeks_B']].to_numpy(), fs)
    c_filt = utils.filter_pos_signal(c_raw, fs)
    c_m = utils.analyse_signal(c_filt, fs, motion_signals=motion)
    
    diff = abs(f_m['bpm'] - c_m['bpm'])
    motion_bpms = [round(f * 60.0, 1) for f in c_m['motion_frequencies_hz']]
    
    return {
        'variant': variant_name,
        'forehead_bpm': round(f_m['bpm'], 1),
        'cheeks_bpm': round(c_m['bpm'], 1),
        'diff_bpm': round(diff, 1),
        'forehead_peak_ratio': round(f_m['peak_ratio'], 2),
        'cheeks_peak_ratio': round(c_m['peak_ratio'], 2),
        'motion_bpms': motion_bpms,
        'regional_agreement': 'PASS' if diff <= 13.0 else 'FAIL',
    }


def main():
    root = Path(__file__).resolve().parent
    stress_dir = root / "stress_results"
    
    # Historical before values from existing pipeline.logs
    before_data = {
        'original': {'forehead': 73.4, 'cheeks': 132.1, 'diff': 59.3},
        'compression': {'forehead': 73.2, 'cheeks': 132.3, 'diff': 59.0},
        'brightness_down': {'forehead': 72.9, 'cheeks': 131.9, 'diff': 59.1},
        'brightness_up': {'forehead': 73.0, 'cheeks': 132.0, 'diff': 59.4},
        'temporal_drop': {'forehead': 74.8, 'cheeks': 132.0, 'diff': 59.4},
    }
    
    records = []
    print("=" * 80)
    print("SIGNALGUARD — BEFORE VS AFTER REGRESSION BENCHMARK")
    print("=" * 80)
    
    variants = ['original', 'compression', 'brightness_down', 'brightness_up', 'temporal_drop']
    for v in variants:
        csv_file = stress_dir / v / "rgb_signals.csv"
        if not csv_file.exists():
            print(f"Skipping {v}: {csv_file} not found")
            continue
        after = evaluate_variant(v, csv_file)
        b = before_data.get(v, {'forehead': 'N/A', 'cheeks': 'N/A', 'diff': 'N/A'})
        
        records.append({
            'Variant': v,
            'Prior Forehead': b['forehead'],
            'Prior Cheeks': b['cheeks'],
            'Prior Diff': b['diff'],
            'Corrected Forehead': after['forehead_bpm'],
            'Corrected Cheeks': after['cheeks_bpm'],
            'Corrected Diff': after['diff_bpm'],
            'Regional Agreement': after['regional_agreement'],
            'Motion Flagged': ", ".join(map(str, after['motion_bpms'])) if after['motion_bpms'] else "None",
        })
        
    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
