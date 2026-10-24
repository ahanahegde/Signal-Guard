"""
SignalGuard Dataset Loader Interface.

Provides a unified interface for loading, organizing, and partitioning face video datasets
(such as FaceForensics++, DFDC, or Celeb-DF) for deepfake classification, while strictly
preventing identity/source-video data leakage between training, validation, and test splits.

Includes an offline synthetic fixture generator for testing and CI.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


LABEL_GENUINE = 0
LABEL_MANIPULATED = 1

NAME_GENUINE = "GENUINE"
NAME_MANIPULATED = "MANIPULATED"

LABEL_TO_NAME = {
    LABEL_GENUINE: NAME_GENUINE,
    LABEL_MANIPULATED: NAME_MANIPULATED,
}

NAME_TO_LABEL = {
    NAME_GENUINE: LABEL_GENUINE,
    NAME_MANIPULATED: LABEL_MANIPULATED,
    "REAL": LABEL_GENUINE,
    "FAKE": LABEL_MANIPULATED,
    "PRISTINE": LABEL_GENUINE,
    "DEEPFAKE": LABEL_MANIPULATED,
}


@dataclass
class VideoRecord:
    """Represents a single labeled video entry."""
    video_path: str
    label: int  # 0 = GENUINE, 1 = MANIPULATED
    label_name: str
    subject_id: str  # Identity / source video ID for grouped partitioning
    condition: str = "default"  # e.g., "c23", "Deepfakes", "original"
    split: str | None = None  # "train", "val", "test", or None
    is_synthetic: bool = False  # Explicit flag: True for synthetic fixtures
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["video_path"] = str(Path(self.video_path).resolve())
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoRecord:
        label = int(data["label"])
        label_name = data.get("label_name", LABEL_TO_NAME.get(label, "UNKNOWN"))
        return cls(
            video_path=data["video_path"],
            label=label,
            label_name=label_name,
            subject_id=str(data.get("subject_id", "unknown")),
            condition=str(data.get("condition", "default")),
            split=data.get("split"),
            is_synthetic=bool(data.get("is_synthetic", False)),
            metadata=data.get("metadata", {}),
        )


class DatasetLoader:
    """
    Handles dataset discovery, manifest parsing, identity-aware splitting,
    and synthetic fixture generation.
    """

    @staticmethod
    def load_from_manifest(manifest_path: str | Path) -> list[VideoRecord]:
        """Load video records from a CSV or JSON manifest file."""
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        records: list[VideoRecord] = []
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("records", data) if isinstance(data, dict) else data
            for item in items:
                records.append(VideoRecord.from_dict(item))
        else:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label_str = row.get("label", "0").strip()
                    if label_str.isdigit():
                        label = int(label_str)
                    else:
                        label = NAME_TO_LABEL.get(label_str.upper(), LABEL_MANIPULATED)

                    meta = {}
                    for k, v in row.items():
                        if k not in {"video_path", "label", "label_name", "subject_id", "condition", "split", "is_synthetic"}:
                            meta[k] = v

                    records.append(
                        VideoRecord(
                            video_path=row["video_path"],
                            label=label,
                            label_name=row.get("label_name", LABEL_TO_NAME[label]),
                            subject_id=row.get("subject_id", "unknown"),
                            condition=row.get("condition", "default"),
                            split=row.get("split") or None,
                            is_synthetic=row.get("is_synthetic", "false").lower() in {"true", "1", "yes"},
                            metadata=meta,
                        )
                    )
        return records

    @staticmethod
    def save_manifest(records: Sequence[VideoRecord], output_path: str | Path) -> Path:
        """Save a list of VideoRecords to CSV or JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.suffix.lower() == ".json":
            with out.open("w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in records], f, indent=2)
        else:
            fieldnames = [
                "video_path",
                "label",
                "label_name",
                "subject_id",
                "condition",
                "split",
                "is_synthetic",
            ]
            with out.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in records:
                    d = r.to_dict()
                    writer.writerow(d)
        return out

    @staticmethod
    def scan_directory(
        root_dir: str | Path,
        real_dirname: str = "real",
        fake_dirname: str = "fake",
        extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    ) -> list[VideoRecord]:
        """
        Scan a structured folder with genuine and manipulated subdirectories.
        Example:
          root_dir/real/*.mp4
          root_dir/fake/*.mp4
        """
        root = Path(root_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"Directory not found: {root}")

        records: list[VideoRecord] = []
        for label, dirname in [(LABEL_GENUINE, real_dirname), (LABEL_MANIPULATED, fake_dirname)]:
            target_dir = root / dirname
            if not target_dir.is_dir():
                continue
            for file_path in target_dir.rglob("*"):
                if file_path.suffix.lower() in extensions and file_path.is_file():
                    # Extract subject_id from stem if separated by underscore (e.g., id01_002 -> id01)
                    stem = file_path.stem
                    subject_id = stem.split("_")[0] if "_" in stem else stem
                    records.append(
                        VideoRecord(
                            video_path=str(file_path.resolve()),
                            label=label,
                            label_name=LABEL_TO_NAME[label],
                            subject_id=subject_id,
                            condition=file_path.parent.name,
                        )
                    )
        return records

    @staticmethod
    def split_by_subject(
        records: Sequence[VideoRecord],
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        random_seed: int = 42,
    ) -> tuple[list[VideoRecord], list[VideoRecord], list[VideoRecord]]:
        """
        Partition records into train, validation, and test splits strictly grouped by subject_id.
        Guarantees that subjects in the test/val sets never appear in the training set (no identity leakage).
        """
        if not records:
            return [], [], []

        rng = random.Random(random_seed)

        # Group records by subject_id
        subject_map: dict[str, list[VideoRecord]] = {}
        for r in records:
            subject_map.setdefault(r.subject_id, []).append(r)

        unique_subjects = list(subject_map.keys())
        rng.shuffle(unique_subjects)

        n_subjects = len(unique_subjects)
        n_test = max(1, int(round(n_subjects * test_ratio))) if test_ratio > 0 else 0
        n_val = max(1, int(round(n_subjects * val_ratio))) if val_ratio > 0 else 0

        # Ensure train gets at least 1 subject if possible
        if n_test + n_val >= n_subjects and n_subjects > 2:
            n_test = 1
            n_val = 1

        test_subjects = set(unique_subjects[:n_test])
        val_subjects = set(unique_subjects[n_test : n_test + n_val])
        train_subjects = set(unique_subjects[n_test + n_val :])

        train_records, val_records, test_records = [], [], []

        for sub, recs in subject_map.items():
            if sub in test_subjects:
                for r in recs:
                    r.split = "test"
                    test_records.append(r)
            elif sub in val_subjects:
                for r in recs:
                    r.split = "val"
                    val_records.append(r)
            else:
                for r in recs:
                    r.split = "train"
                    train_records.append(r)

        return train_records, val_records, test_records

    @staticmethod
    def generate_synthetic_fixture(
        output_dir: str | Path,
        num_subjects: int = 6,
        clips_per_subject: int = 4,
        fps: float = 16.0,
        duration_seconds: float = 2.0,
        width: int = 320,
        height: int = 240,
    ) -> list[VideoRecord]:
        """
        Generate a lightweight synthetic validation fixture with clearly labeled
        genuine and manipulated video records.

        NOTICE: These videos are synthetic mathematical test fixtures designed for unit testing
        and pipeline validation only. They are NOT real research datasets.
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        records: list[VideoRecord] = []
        n_frames = max(16, int(round(duration_seconds * fps)))

        for sub_idx in range(num_subjects):
            sub_id = f"synthetic_sub_{sub_idx:03d}"
            base_color = np.array([
                110 + (sub_idx * 15) % 80,
                130 + (sub_idx * 12) % 80,
                160 + (sub_idx * 17) % 80,
            ], dtype=np.uint8)

            for clip_idx in range(clips_per_subject):
                # Alternate genuine vs manipulated per subject
                is_manipulated = (clip_idx % 2 == 1)
                label = LABEL_MANIPULATED if is_manipulated else LABEL_GENUINE
                label_name = LABEL_TO_NAME[label]
                condition = "artifact_synthesis" if is_manipulated else "pristine_synthesis"

                filename = f"{sub_id}_clip_{clip_idx:02d}_{label_name.lower()}.mp4"
                file_path = out_root / filename

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

                center_x, center_y = width // 2, height // 2
                radius_x, radius_y = width // 4, height // 3

                for frame_idx in range(n_frames):
                    t = frame_idx / fps
                    # Synthetic face frame: neutral background
                    frame = np.full((height, width, 3), 40, dtype=np.uint8)

                    # Draw an oval simulating a face
                    cv2.ellipse(
                        frame,
                        (center_x, center_y),
                        (radius_x, radius_y),
                        0,
                        0,
                        360,
                        base_color.tolist(),
                        -1,
                    )

                    if is_manipulated:
                        # Introduce synthetic deepfake-like artifacts:
                        # 1. High-frequency boundary blur / checkerboard noise in center
                        noise = np.random.normal(0, 18, (radius_y, radius_x, 3)).astype(np.int16)
                        patch = frame[
                            center_y - radius_y // 2 : center_y + radius_y // 2,
                            center_x - radius_x // 2 : center_x + radius_x // 2,
                        ].astype(np.int16)
                        patch = np.clip(patch + noise, 0, 255).astype(np.uint8)
                        frame[
                            center_y - radius_y // 2 : center_y + radius_y // 2,
                            center_x - radius_x // 2 : center_x + radius_x // 2,
                        ] = patch

                        # 2. Boundary discontinuity ring
                        cv2.ellipse(
                            frame,
                            (center_x, center_y),
                            (radius_x - 4, radius_y - 4),
                            0,
                            0,
                            360,
                            (25, 25, 25),
                            2,
                        )
                    else:
                        # Genuine: smooth natural shading + subtle physiological pulse in green channel
                        pulse = np.sin(2.0 * np.pi * 1.25 * t)  # ~75 BPM
                        green_shift = int(round(pulse * 3.0))
                        g_chan = np.clip(frame[:, :, 1].astype(int) + green_shift, 0, 255).astype(np.uint8)
                        frame[:, :, 1] = g_chan

                    writer.write(frame)

                writer.release()

                records.append(
                    VideoRecord(
                        video_path=str(file_path.resolve()),
                        label=label,
                        label_name=label_name,
                        subject_id=sub_id,
                        condition=condition,
                        is_synthetic=True,
                        metadata={
                            "frames": n_frames,
                            "fps": fps,
                            "resolution": f"{width}x{height}",
                            "fixture_type": "synthetic_mathematical_toy",
                        },
                    )
                )

        manifest_file = out_root / "manifest.csv"
        DatasetLoader.save_manifest(records, manifest_file)
        return records


# Convenience module-level aliases
load_from_manifest = DatasetLoader.load_from_manifest
save_manifest = DatasetLoader.save_manifest
scan_directory = DatasetLoader.scan_directory
split_by_subject = DatasetLoader.split_by_subject
generate_synthetic_fixture = DatasetLoader.generate_synthetic_fixture


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="SignalGuard Dataset Loader and Fixture Generator")
    parser.add_argument("--generate-fixture", type=str, help="Directory to output synthetic test fixture")
    parser.add_argument("--num-subjects", type=int, default=8, help="Number of synthetic subjects to generate")
    parser.add_argument("--clips-per-subject", type=int, default=4, help="Clips per synthetic subject")
    parser.add_argument("--scan-dir", type=str, help="Directory to scan for real/fake subdirectories")
    parser.add_argument("--output-manifest", type=str, help="Path to write manifest CSV")
    parser.add_argument("--split", action="store_true", help="Automatically partition records by subject into train/val/test")

    args = parser.parse_args()

    if args.generate_fixture:
        out_dir = Path(args.generate_fixture)
        print(f"Generating synthetic fixture with {args.num_subjects} subjects in: {out_dir}")
        records = generate_synthetic_fixture(
            out_dir,
            num_subjects=args.num_subjects,
            clips_per_subject=args.clips_per_subject,
        )
        if args.split:
            train_recs, val_recs, test_recs = split_by_subject(records, val_ratio=0.25, test_ratio=0.25)
            all_recs = train_recs + val_recs + test_recs
            manifest_p = args.output_manifest or str(out_dir / "manifest.csv")
            save_manifest(all_recs, manifest_p)
            print(f"Saved split manifest with {len(all_recs)} records (train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)})")
        else:
            print(f"Fixture created with {len(records)} records. Manifest: {out_dir / 'manifest.csv'}")
        return 0

    if args.scan_dir:
        records = scan_directory(args.scan_dir)
        print(f"Discovered {len(records)} video records in {args.scan_dir}")
        if args.split:
            train_recs, val_recs, test_recs = split_by_subject(records)
            records = train_recs + val_recs + test_recs
        out_p = args.output_manifest or "dataset_manifest.csv"
        save_manifest(records, out_p)
        print(f"Manifest written to: {out_p}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

