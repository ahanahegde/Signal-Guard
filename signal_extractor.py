import sys
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SIGNALGUARD - ROI TRACKING AND 15-REGION RGB EXTRACTION
# ============================================================

MODEL_PATH = Path("face_landmarker.task")
OUTPUT_CSV = Path("rgb_signals.csv")
OUTPUT_PLOT = Path("rgb_signals.png")
OUTPUT_PREVIEW = Path("roi_preview.jpg")
OUTPUT_PREVIEW_15 = Path("roi_preview_15regions.jpg")

# Experimental quality settings; calibrate on test recordings.
MIN_FOREHEAD_PIXELS = 120
MIN_CHEEK_PIXELS = 80
MIN_SUBREGION_PIXELS = 30
MAX_SATURATED_FRACTION = 0.25
SATURATION_LOW = 5
SATURATION_HIGH = 250

# ------------------------------------------------------------
# 15 ANATOMICAL FACIAL SUB-REGIONS (MediaPipe FaceMesh Mapping)
# ------------------------------------------------------------
ANATOMICAL_SUBREGIONS = {
    "medial_forehead": [10, 151, 9, 8, 107, 336, 108, 337, 69, 299],
    "glabella": [8, 9, 107, 336, 168, 6, 197, 193, 417],
    "left_lateral_forehead": [109, 67, 103, 54, 21, 162, 127, 69, 108],
    "right_lateral_forehead": [338, 297, 332, 284, 251, 389, 356, 299, 337],
    "left_temporal": [54, 21, 162, 127, 234, 93, 137],
    "right_temporal": [284, 251, 389, 356, 454, 323, 366],
    "nasal_dorsum": [6, 197, 195, 5, 4, 196, 419, 168, 122, 351],
    "left_malar": [116, 117, 118, 101, 50, 187, 123, 119, 120, 100, 126, 142],
    "right_malar": [345, 346, 347, 330, 280, 411, 352, 348, 349, 329, 355, 371],
    "left_submalar": [205, 206, 207, 213, 192, 138, 137, 227, 187, 123, 50],
    "right_submalar": [425, 426, 427, 433, 416, 367, 366, 447, 411, 352, 280],
    "philtrum": [164, 0, 37, 267, 39, 269, 185, 409, 2, 94],
    "mentum": [152, 148, 176, 149, 150, 175, 199, 200, 201, 208, 428, 421, 377, 400, 378, 379],
    "left_mandibular": [172, 136, 150, 149, 176, 58, 132, 93, 234, 138],
    "right_mandibular": [397, 365, 379, 378, 400, 288, 361, 323, 454, 367],
}

# Color palette for 15-region visual preview (BGR)
SUBREGION_COLORS = {
    "medial_forehead": (71, 99, 255),       # Tomato
    "glabella": (255, 144, 30),             # Dodger Blue
    "left_lateral_forehead": (113, 179, 60), # Medium Sea Green
    "right_lateral_forehead": (0, 215, 255), # Gold
    "left_temporal": (226, 43, 138),        # Blue Violet
    "right_temporal": (180, 105, 255),      # Hot Pink
    "nasal_dorsum": (255, 255, 0),          # Cyan
    "left_malar": (0, 140, 255),            # Dark Orange
    "right_malar": (50, 205, 50),           # Lime Green
    "left_submalar": (180, 130, 70),        # Steel Blue
    "right_submalar": (214, 112, 218),      # Orchid
    "philtrum": (140, 230, 240),            # Khaki
    "mentum": (122, 160, 255),              # Light Salmon
    "left_mandibular": (237, 149, 100),     # Cornflower Blue
    "right_mandibular": (144, 238, 144),    # Light Green
}

# ------------------------------------------------------------
# AGGREGATE REGIONS (For backward compatibility with POS)
# ------------------------------------------------------------
FOREHEAD_IDS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    127, 162, 21, 54, 103, 67, 109
]

CHEEK_A_IDS = [
    50, 101, 118, 119, 120, 121, 126, 142,
    36, 205, 206, 207, 213, 187, 123
]

CHEEK_B_IDS = [
    280, 330, 347, 348, 349, 350, 355, 371,
    266, 425, 426, 427, 436, 411, 352
]

FACE_OVAL_IDS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109
]

LEFT_EYE_IDS = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_IDS = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

LEFT_BROW_IDS = [70, 63, 105, 66, 107]
RIGHT_BROW_IDS = [336, 296, 334, 293, 300]

MOUTH_IDS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185
]

NOSE_IDS = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275]


# ============================================================
# MASK UTILITIES
# ============================================================

def landmark_points(landmarks, ids, width, height):
    """Convert normalized landmarks to image pixel coordinates."""
    if not ids or max(ids) >= len(landmarks):
        return None

    points = []
    for idx in ids:
        lm = landmarks[idx]
        x = int(round(lm.x * width))
        y = int(round(lm.y * height))
        points.append([x, y])

    return np.asarray(points, dtype=np.int32)


def make_hull_mask(landmarks, ids, width, height):
    """Create a filled convex-hull mask from selected landmarks."""
    points = landmark_points(landmarks, ids, width, height)
    mask = np.zeros((height, width), dtype=np.uint8)

    if points is None or len(points) < 3:
        return mask

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def make_exclusion_mask(landmarks, width, height, feature_groups, kernel_size):
    """Build expanded masks for facial features to exclude."""
    exclusion = np.zeros((height, width), dtype=np.uint8)

    for ids in feature_groups:
        feature_mask = make_hull_mask(landmarks, ids, width, height)
        exclusion = cv2.bitwise_or(exclusion, feature_mask)

    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(exclusion, kernel, iterations=1)


def clean_roi(mask, face_mask, exclusion_mask):
    """Restrict ROI to the face, exclude features, and erode its edge."""
    result = cv2.bitwise_and(mask, face_mask)
    if exclusion_mask is not None:
        result[exclusion_mask > 0] = 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    result = cv2.erode(result, kernel, iterations=1)
    return result


# ============================================================
# RGB SAMPLING AND QUALITY METRICS
# ============================================================

def sample_region(frame_rgb, mask, minimum_pixels):
    """
    Measure mean RGB for a region.
    Return NaN RGB values when region area is too small or
    the fraction of saturated pixels exceeds the configured limit.
    """
    selected = frame_rgb[mask > 0]
    pixel_count = len(selected)

    if pixel_count == 0:
        return {
            "rgb": [np.nan, np.nan, np.nan],
            "pixels": 0,
            "saturated_fraction": np.nan,
            "brightness": np.nan,
            "valid": False,
        }

    saturated = np.any(
        (selected <= SATURATION_LOW) |
        (selected >= SATURATION_HIGH),
        axis=1,
    )

    saturated_fraction = float(np.mean(saturated))
    brightness = float(np.mean(selected))

    valid = (
        pixel_count >= minimum_pixels
        and saturated_fraction <= MAX_SATURATED_FRACTION
    )

    if valid:
        mean_rgb = np.mean(selected, axis=0).tolist()
    else:
        mean_rgb = [np.nan, np.nan, np.nan]

    return {
        "rgb": mean_rgb,
        "pixels": int(pixel_count),
        "saturated_fraction": saturated_fraction,
        "brightness": brightness,
        "valid": valid,
    }


def add_sample_to_row(row, prefix, sample):
    row[f"{prefix}_R"] = sample["rgb"][0]
    row[f"{prefix}_G"] = sample["rgb"][1]
    row[f"{prefix}_B"] = sample["rgb"][2]
    row[f"{prefix}_roi_pixels"] = sample["pixels"]
    row[f"{prefix}_saturated_fraction"] = sample["saturated_fraction"]
    row[f"{prefix}_brightness"] = sample["brightness"]
    row[f"{prefix}_valid"] = int(sample["valid"])


# ============================================================
# INPUT & INITIALIZATION
# ============================================================

if not MODEL_PATH.exists():
    raise SystemExit(
        "ERROR: face_landmarker.task not found in the project folder."
    )

if len(sys.argv) > 1:
    video_path = sys.argv[1].strip().strip('"')
else:
    video_path = input("Enter video path: ").strip().strip('"')

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise SystemExit(f"ERROR: Could not open video at '{video_path}'.")

fps = video.get(cv2.CAP_PROP_FPS)

if not np.isfinite(fps) or fps <= 0:
    video.release()
    raise SystemExit("ERROR: Video has invalid FPS metadata.")

total_frames_reported = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision

options = Vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH.resolve())),
    running_mode=Vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# PROCESS VIDEO
# ============================================================

records = []
frame_index = 0

frames_with_landmarks = 0
frames_with_valid_regions = 0

previous_center = None
previous_face_width = None

preview_saved = False

ALL_REGION_PREFIXES = [
    "forehead", "left_cheek", "right_cheek", "cheeks"
] + list(ANATOMICAL_SUBREGIONS.keys())

with Vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame_bgr = video.read()
        if not ok:
            break

        height, width = frame_bgr.shape[:2]

        timestamp = frame_index / fps
        timestamp_ms = round(timestamp * 1000)
        frame_index += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )

        result = landmarker.detect_for_video(image, timestamp_ms)

        row = {
            "time_seconds": timestamp,
            "face_detected": 0,
            "valid_region_samples": 0,
            "valid_subregions_count": 0,
            "face_center_x_norm": np.nan,
            "face_center_y_norm": np.nan,
            "face_width_norm": np.nan,
            "face_height_norm": np.nan,
            "face_center_shift_frame_fraction": np.nan,
            "face_width_change_pct": np.nan,
        }

        for prefix in ALL_REGION_PREFIXES:
            row[f"{prefix}_R"] = np.nan
            row[f"{prefix}_G"] = np.nan
            row[f"{prefix}_B"] = np.nan
            row[f"{prefix}_roi_pixels"] = 0
            row[f"{prefix}_saturated_fraction"] = np.nan
            row[f"{prefix}_brightness"] = np.nan
            row[f"{prefix}_valid"] = 0

        if not result.face_landmarks:
            previous_center = None
            previous_face_width = None
            records.append(row)
            continue

        frames_with_landmarks += 1
        landmarks = result.face_landmarks[0]

        # Face bounding box based on landmark coordinates.
        coords = np.asarray([
            [lm.x * width, lm.y * height]
            for lm in landmarks
        ], dtype=float)

        x_min = float(np.clip(coords[:, 0].min(), 0, width - 1))
        x_max = float(np.clip(coords[:, 0].max(), 0, width - 1))
        y_min = float(np.clip(coords[:, 1].min(), 0, height - 1))
        y_max = float(np.clip(coords[:, 1].max(), 0, height - 1))

        face_width = max(1.0, x_max - x_min)
        face_height = max(1.0, y_max - y_min)

        center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
        ])

        row["face_detected"] = 1
        row["face_center_x_norm"] = center[0] / width
        row["face_center_y_norm"] = center[1] / height
        row["face_width_norm"] = face_width / width
        row["face_height_norm"] = face_height / height

        if previous_center is not None:
            movement = np.linalg.norm(center - previous_center)
            row["face_center_shift_frame_fraction"] = (
                movement / np.hypot(width, height)
            )

        if previous_face_width is not None:
            row["face_width_change_pct"] = (
                abs(face_width - previous_face_width)
                / max(previous_face_width, 1.0)
                * 100.0
            )

        previous_center = center
        previous_face_width = face_width

        # Face boundary mask
        face_mask = make_hull_mask(landmarks, FACE_OVAL_IDS, width, height)

        # Facial feature exclusion masks scaled to face size
        exclusion_kernel = max(5, round(face_width * 0.025))
        eye_exclusion = make_exclusion_mask(
            landmarks, width, height,
            [LEFT_EYE_IDS, RIGHT_EYE_IDS],
            exclusion_kernel,
        )

        brow_kernel = max(11, round(face_width * 0.07))
        brow_exclusion = make_exclusion_mask(
            landmarks, width, height,
            [LEFT_BROW_IDS, RIGHT_BROW_IDS],
            brow_kernel,
        )

        mouth_exclusion = make_exclusion_mask(
            landmarks, width, height,
            [MOUTH_IDS],
            exclusion_kernel,
        )

        nose_exclusion = make_exclusion_mask(
            landmarks, width, height,
            [NOSE_IDS],
            exclusion_kernel,
        )

        upper_exclusions = cv2.bitwise_or(eye_exclusion, brow_exclusion)
        lower_exclusions = mouth_exclusion
        mid_exclusions = cv2.bitwise_or(eye_exclusion, mouth_exclusion)
        mid_exclusions = cv2.bitwise_or(mid_exclusions, nose_exclusion)

        # --------------------------------------------------------
        # 1. SAMPLE 15 ANATOMICAL SUB-REGIONS
        # --------------------------------------------------------
        subregion_masks = {}
        subregion_samples = {}
        subregions_valid = 0

        for reg_name, reg_ids in ANATOMICAL_SUBREGIONS.items():
            base_mask = make_hull_mask(landmarks, reg_ids, width, height)

            # Determine appropriate exclusion zone for the sub-region
            if "forehead" in reg_name or "temporal" in reg_name or reg_name == "glabella":
                excl = upper_exclusions
            elif "mandibular" in reg_name or reg_name in ["philtrum", "mentum"]:
                excl = lower_exclusions
            elif "malar" in reg_name:
                excl = mid_exclusions
            else:
                excl = eye_exclusion

            clean_mask = clean_roi(base_mask, face_mask, excl)
            sample = sample_region(frame_rgb, clean_mask, MIN_SUBREGION_PIXELS)

            subregion_masks[reg_name] = clean_mask
            subregion_samples[reg_name] = sample
            add_sample_to_row(row, reg_name, sample)

            if sample["valid"]:
                subregions_valid += 1

        row["valid_subregions_count"] = subregions_valid

        # --------------------------------------------------------
        # 2. SAMPLE AGGREGATE REGIONS (For POS Backwards Compatibility)
        # --------------------------------------------------------
        forehead_base = make_hull_mask(landmarks, FOREHEAD_IDS, width, height)
        forehead_mask = clean_roi(forehead_base, face_mask, upper_exclusions)

        cheek_a_base = make_hull_mask(landmarks, CHEEK_A_IDS, width, height)
        cheek_b_base = make_hull_mask(landmarks, CHEEK_B_IDS, width, height)

        cheek_a_mask = clean_roi(cheek_a_base, face_mask, mid_exclusions)
        cheek_b_mask = clean_roi(cheek_b_base, face_mask, mid_exclusions)

        forehead_sample = sample_region(frame_rgb, forehead_mask, MIN_FOREHEAD_PIXELS)
        cheek_a_sample = sample_region(frame_rgb, cheek_a_mask, MIN_CHEEK_PIXELS)
        cheek_b_sample = sample_region(frame_rgb, cheek_b_mask, MIN_CHEEK_PIXELS)

        add_sample_to_row(row, "forehead", forehead_sample)
        add_sample_to_row(row, "left_cheek", cheek_a_sample)
        add_sample_to_row(row, "right_cheek", cheek_b_sample)

        # Combined cheek region
        if cheek_a_sample["valid"] and cheek_b_sample["valid"]:
            combined_cheeks_mask = cv2.bitwise_or(cheek_a_mask, cheek_b_mask)
            combined_cheeks = sample_region(
                frame_rgb, combined_cheeks_mask, 2 * MIN_CHEEK_PIXELS
            )
        elif cheek_a_sample["valid"]:
            combined_cheeks = cheek_a_sample
        elif cheek_b_sample["valid"]:
            combined_cheeks = cheek_b_sample
        else:
            combined_cheeks = {
                "rgb": [np.nan, np.nan, np.nan],
                "pixels": cheek_a_sample["pixels"] + cheek_b_sample["pixels"],
                "saturated_fraction": np.nan,
                "brightness": np.nan,
                "valid": False,
            }

        add_sample_to_row(row, "cheeks", combined_cheeks)

        # Usability flag for POS
        row["valid_region_samples"] = int(
            forehead_sample["valid"] and combined_cheeks["valid"]
        )

        if row["valid_region_samples"]:
            frames_with_valid_regions += 1

        # --------------------------------------------------------
        # 3. SAVE VISUAL PREVIEWS
        # --------------------------------------------------------
        if not preview_saved and forehead_sample["pixels"] > 0:
            # 3A. Classic 3-ROI Preview (Forehead + 2 Cheeks)
            preview = frame_bgr.copy()
            overlay = preview.copy()
            overlay[forehead_mask > 0] = (0, 255, 255)
            overlay[cheek_a_mask > 0] = (0, 255, 0)
            overlay[cheek_b_mask > 0] = (255, 0, 255)
            preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)

            cv2.putText(
                preview, "Forehead", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                preview, "Cheek A", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                preview, "Cheek B", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
            )
            cv2.imwrite(str(OUTPUT_PREVIEW), preview)

            # 3B. Rich 15-Region Anatomical Preview with Labels & Polylines
            preview_15 = frame_bgr.copy()
            overlay_15 = frame_bgr.copy()

            for reg_name, reg_ids in ANATOMICAL_SUBREGIONS.items():
                color = SUBREGION_COLORS.get(reg_name, (0, 255, 255))
                pts = []
                for pt_idx in reg_ids:
                    lm = landmarks[pt_idx]
                    pts.append([int(round(lm.x * width)), int(round(lm.y * height))])
                pts = np.asarray(pts, dtype=np.int32)
                hull = cv2.convexHull(pts)

                cv2.fillConvexPoly(overlay_15, hull, color)
                cv2.polylines(preview_15, [hull], True, (255, 255, 255), 1)

                moments = cv2.moments(hull)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m11"] / moments["m00"])
                    cv2.circle(preview_15, (cx, cy), 3, (255, 255, 255), -1)
                    short_name = reg_name.replace("_", " ").title()
                    cv2.putText(
                        preview_15, short_name, (cx - 28, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2
                    )
                    cv2.putText(
                        preview_15, short_name, (cx - 28, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1
                    )

            blended_15 = cv2.addWeighted(overlay_15, 0.45, preview_15, 0.55, 0)
            cv2.imwrite(str(OUTPUT_PREVIEW_15), blended_15)

            preview_saved = True

        records.append(row)

video.release()

if not records:
    raise SystemExit("ERROR: No frames were processed.")


# ============================================================
# SAVE RESULTS & METRICS SUMMARY
# ============================================================

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("SIGNALGUARD - 15-REGION ROI EXTRACTION SUMMARY")
print("=" * 60)
print(f"Frames processed: {len(df)}")
print(f"Video FPS metadata: {fps:.3f}")
print(f"Duration: {len(df) / fps:.2f} seconds")
print(f"Frames with landmarks: {frames_with_landmarks}")
print(f"Frames with valid forehead and cheek ROIs: {frames_with_valid_regions}")
print(f"Face detection rate: {frames_with_landmarks / len(df) * 100:.2f}%")
print(f"Valid ROI sampling coverage: {frames_with_valid_regions / len(df) * 100:.2f}%")
if "valid_subregions_count" in df.columns:
    print(f"Mean valid sub-regions per frame: {df['valid_subregions_count'].mean():.1f} / 15")

print("\nAGGREGATE REGIONS:")
for region in ["forehead", "left_cheek", "right_cheek", "cheeks"]:
    valid_col = f"{region}_valid"
    pixel_col = f"{region}_roi_pixels"
    saturation_col = f"{region}_saturated_fraction"

    valid_rate = df[valid_col].mean() * 100
    pixel_values = df.loc[df[valid_col] == 1, pixel_col]
    saturation_values = df.loc[df[valid_col] == 1, saturation_col].dropna()

    median_pixels = float(pixel_values.median()) if len(pixel_values) else 0.0
    median_saturation = (
        float(saturation_values.median() * 100) if len(saturation_values) else float("nan")
    )
    print(f"  {region:12s}: Coverage {valid_rate:5.1f}% | Median Pixels {median_pixels:5.0f} | Median Saturation {median_saturation:4.1f}%")

print("\n15 ANATOMICAL SUB-REGIONS:")
for reg_name in ANATOMICAL_SUBREGIONS.keys():
    valid_col = f"{reg_name}_valid"
    pixel_col = f"{reg_name}_roi_pixels"
    saturation_col = f"{reg_name}_saturated_fraction"

    valid_rate = df[valid_col].mean() * 100
    pixel_values = df.loc[df[valid_col] == 1, pixel_col]
    saturation_values = df.loc[df[valid_col] == 1, saturation_col].dropna()

    median_pixels = float(pixel_values.median()) if len(pixel_values) else 0.0
    median_saturation = (
        float(saturation_values.median() * 100) if len(saturation_values) else float("nan")
    )
    print(f"  {reg_name:24s}: Coverage {valid_rate:5.1f}% | Median Pixels {median_pixels:5.0f} | Median Saturation {median_saturation:4.1f}%")

center_shift = df["face_center_shift_frame_fraction"].dropna()
scale_change = df["face_width_change_pct"].dropna()

print("\nTRACKING DIAGNOSTICS")
if len(center_shift):
    print(f"  Median face-center movement: {center_shift.median():.5f} (frac diagonal)")
else:
    print("  Face-center movement: unavailable")

if len(scale_change):
    print(f"  Median face-width change: {scale_change.median():.2f}%")
else:
    print("  Face-width change: unavailable")

print(f"\nOutputs saved:")
print(f"  - RGB CSV: {OUTPUT_CSV}")
if preview_saved:
    print(f"  - Standard ROI preview: {OUTPUT_PREVIEW}")
    print(f"  - 15-Region preview:    {OUTPUT_PREVIEW_15}")


# ============================================================
# PLOT RGB MEASUREMENTS
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
times = df["time_seconds"]

for channel in ["R", "G", "B"]:
    axes[0].plot(times, df[f"forehead_{channel}"], label=f"Forehead {channel}", linewidth=1)
    axes[1].plot(times, df[f"cheeks_{channel}"], label=f"Cheeks {channel}", linewidth=1)

# Third plot: Green channel across representative sub-regions (pulse is strongest in G)
rep_subregions = ["medial_forehead", "glabella", "left_malar", "right_malar", "mentum"]
for subreg in rep_subregions:
    if f"{subreg}_G" in df.columns:
        axes[2].plot(times, df[f"{subreg}_G"], label=subreg.replace("_", " ").title(), linewidth=1)

axes[0].set_title("Aggregate Forehead RGB Signals")
axes[1].set_title("Aggregate Cheeks RGB Signals")
axes[2].set_title("15 Anatomical Sub-Regions - Green Channel PPG Profiles")
axes[2].set_xlabel("Time (seconds)")

for ax in axes:
    ax.set_ylabel("Mean Intensity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"  - RGB plot:             {OUTPUT_PLOT}")
plt.close(fig)
