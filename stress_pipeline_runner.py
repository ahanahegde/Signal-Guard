from __future__ import annotations

"""
SignalGuard stress pipeline runner.

Place this file in the same folder as:
  signal_extractor.py
  pos_extractor.py
  eligibility_gate.py
  temporal_consistency.py
  video_forensics.py
  signalguard_utils.py
  face_landmarker.task

Usage:
  python stress_pipeline_runner.py "C:\\path\\to\\video.mp4"

Options:
  --no-generate   Reuse MP4 variants already in stress_outputs
  --skip-original Do not run the original video
  --only LABEL    Process only one variant (repeatable)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRESS_DIR = ROOT / "stress_outputs"
RESULTS_DIR = ROOT / "stress_results"

PIPELINE_SCRIPTS = (
    "signal_extractor.py",
    "pos_extractor.py",
    "eligibility_gate.py",
    "temporal_consistency.py",
    "video_forensics.py",
)

ARTIFACTS = (
    "rgb_signals.csv",
    "rgb_signals.png",
    "roi_preview.jpg",
    "pos_signals.csv",
    "pos_signals.png",
    "window_analysis.csv",
    "reliability_report.json",
    "temporal_consistency.csv",
    "temporal_consistency.json",
    "video_forensics.csv",
    "video_forensics.json",
)

VARIANT_NAMES = (
    "compression",
    "brightness_down",
    "brightness_up",
    "temporal_drop",
)

VIDEO_SCRIPTS = {
    "temporal_consistency.py",
    "video_forensics.py",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "video"


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def check_project_files() -> list[str]:
    required = [
        *PIPELINE_SCRIPTS, "signalguard_utils.py", "face_landmarker.task"
    ]
    return [name for name in required if not (ROOT / name).is_file()]


def make_variant(source: Path, destination: Path, kind: str) -> None:
    """Create a stress-test copy; the source video is never modified."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dimensions: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create MP4 output: {destination}. "
            "Try updating/reinstalling opencv-python."
        )

    frame_index = 0
    written = 0
    previous_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if kind == "brightness_down":
                frame = cv2.convertScaleAbs(frame, alpha=0.82, beta=-12)
            elif kind == "brightness_up":
                frame = cv2.convertScaleAbs(frame, alpha=1.08, beta=12)
            elif kind == "temporal_drop":
                # Simulate dropped frames by duplicating the preceding frame
                # at regular intervals. This preserves clip duration and FPS.
                if frame_index % 5 == 4 and previous_frame is not None:
                    frame = previous_frame.copy()
                else:
                    previous_frame = frame.copy()
            elif kind == "compression":
                # Re-encoding at a lower-resolution intermediate and back
                # introduces mild spatial compression/resampling artifacts.
                small_w = max(2, (width * 3 // 4) // 2 * 2)
                small_h = max(2, (height * 3 // 4) // 2 * 2)
                small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
                frame = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

            writer.write(frame)
            written += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if written == 0 or not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"Variant generation produced no video: {destination}")

    print(
        f"Created {kind}: {destination.name} "
        f"({written}/{frame_count or written} frames, {fps:.2f} FPS)"
    )


def generate_variants(original: Path) -> list[tuple[str, Path]]:
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    variants: list[tuple[str, Path]] = []
    manifest = {
        "original_video": str(original.resolve()),
        "generated_at": timestamp(),
        "note": (
            "Stress variants are robustness probes, not authenticity labels. "
            "Eligibility verdicts describe signal quality only."
        ),
        "variants": [],
    }

    for kind in VARIANT_NAMES:
        target = STRESS_DIR / f"{kind}.mp4"
        make_variant(original, target, kind)
        variants.append((kind, target.resolve()))
        manifest["variants"].append({
            "name": kind,
            "path": str(target.resolve()),
            "transformation": {
                "compression": "Downscale to 75% and resize back",
                "brightness_down": "Brightness/contrast adjustment alpha=0.82, beta=-12",
                "brightness_up": "Brightness/contrast adjustment alpha=1.08, beta=12",
                "temporal_drop": "Duplicate every fifth frame to simulate temporal drops",
            }[kind],
        })

    (STRESS_DIR / "stress_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return variants


def run_script(name: str, video: Path | None = None) -> tuple[int, str]:
    script_path = ROOT / name

    video_scripts = {
        "temporal_consistency.py",
        "video_forensics.py",
    }

    if name in video_scripts:
        if video is None:
            return 2, "ERROR: Video path is required."
        command = [
            sys.executable, "-u", str(script_path),
            str(video.resolve()),
        ]
        input_text = None
    elif name == "eligibility_gate.py":
        command = [
            sys.executable, "-u", str(script_path),
            "--out-json", "reliability_report.json",
        ]
        input_text = None
    else:
        command = [sys.executable, "-u", str(script_path)]
        input_text = (
            f"{video.resolve()}\n"
            if name == "signal_extractor.py" and video
            else None
        )

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            env=env,
            timeout=900,
        )
        return result.returncode, result.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, output + "\nERROR: script timed out.\n"


def archive_outputs(folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ARTIFACTS:
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, folder / name)
            copied.append(name)
    return copied


def process_video(video: Path, label: str) -> dict:
    outdir = RESULTS_DIR / safe_name(label)
    outdir.mkdir(parents=True, exist_ok=True)

    record = {
        "variant": label,
        "video_path": str(video.resolve()),
        "status": "FAILED",
        "gate_verdict": "UNKNOWN",
        "steps": {},
        "started_at": timestamp(),
    }
    logs: list[str] = []
    print(f"\n{'=' * 72}\nPROCESSING: {label}\nVIDEO: {video}\n{'=' * 72}")

    # Remove previous run's outputs so a failed stage cannot accidentally
    # archive stale artifacts from another video.
    for name in ARTIFACTS:
        try:
            (ROOT / name).unlink(missing_ok=True)
        except OSError:
            pass

    for name in PIPELINE_SCRIPTS:
        code, output = run_script(name, video if (name == "signal_extractor.py" or name in VIDEO_SCRIPTS) else None)
        print(f"\n--- {name} (exit {code}) ---\n{output}")
        logs.append(f"\n{'=' * 24} {name} | exit={code} {'=' * 24}\n{output}\n")
        record["steps"][name] = {"exit_code": code, "success": code == 0}

        if code != 0:
            record["failed_step"] = name
            break

        if name == "eligibility_gate.py":
            matches = re.findall(r"\b(ELIGIBLE|INCONCLUSIVE)\b", output.upper())
            if matches:
                record["gate_verdict"] = matches[-1]

    record["archived_files"] = archive_outputs(outdir)
    (outdir / "pipeline.log").write_text("".join(logs), encoding="utf-8")
    record["finished_at"] = timestamp()

    if (
        len(record["steps"]) == len(PIPELINE_SCRIPTS)
        and all(step["success"] for step in record["steps"].values())
    ):
        record["status"] = "COMPLETED"

    (outdir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        f"\nSaved results: {outdir}\n"
        f"Status: {record['status']} | Gate verdict: {record['gate_verdict']}"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SignalGuard on an original video and generated stress-test variants."
    )
    parser.add_argument("original_video", help="Path to the original MP4 video")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Reuse existing MP4 variants in stress_outputs instead of generating new ones",
    )
    parser.add_argument(
        "--skip-original",
        action="store_true",
        help="Process only stress-test variants, not the original video",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=("original", *VARIANT_NAMES),
        help="Run only this variant (can be repeated)",
    )
    args = parser.parse_args()

    original = Path(args.original_video.strip().strip('"')).expanduser()
    if not original.is_absolute():
        original = (Path.cwd() / original).resolve()
    if not original.is_file():
        print(f"ERROR: Original video does not exist:\n  {original}")
        return 2

    missing = check_project_files()
    if missing:
        print("ERROR: Required project files are missing from the runner's folder:")
        for name in missing:
            print(f"  - {ROOT / name}")
        print(
            "\nPut stress_pipeline_runner.py beside these files and run it again. "
            "This runner does not invent or replace your signal-processing modules."
        )
        return 2

    if args.no_generate:
        variants = []
        for name in VARIANT_NAMES:
            candidate = STRESS_DIR / f"{name}.mp4"
            if candidate.is_file():
                variants.append((name, candidate.resolve()))
        if not variants:
            print(f"ERROR: No known stress variants found in {STRESS_DIR}")
            print("Expected: " + ", ".join(f"{name}.mp4" for name in VARIANT_NAMES))
            return 2
    else:
        try:
            variants = generate_variants(original)
        except Exception as error:
            print(f"\nERROR generating stress variants: {type(error).__name__}: {error}")
            return 2

    jobs: list[tuple[str, Path]] = []
    if not args.skip_original:
        jobs.append(("original", original.resolve()))
    jobs.extend(variants)

    if args.only:
        selected = set(args.only)
        jobs = [job for job in jobs if job[0] in selected]
    if not jobs:
        print("ERROR: No variants selected to process.")
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = [process_video(path, label) for label, path in jobs]

    summary = [
        {
            "variant": result["variant"],
            "status": result["status"],
            "gate_verdict": result["gate_verdict"],
            "failed_step": result.get("failed_step", ""),
            "result_folder": str((RESULTS_DIR / safe_name(result["variant"])).resolve()),
        }
        for result in results
    ]
    (RESULTS_DIR / "stress_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    try:
        import csv
        with (RESULTS_DIR / "stress_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("variant", "status", "gate_verdict", "failed_step", "result_folder"),
            )
            writer.writeheader()
            writer.writerows(summary)
    except Exception as error:
        print(f"WARNING: Could not write CSV summary: {error}")

    print(f"\n{'=' * 72}\nSTRESS PIPELINE SUMMARY\n{'=' * 72}")
    for row in summary:
        print(
            f"{row['variant']:<22} {row['status']:<10} "
            f"verdict={row['gate_verdict']:<12} failed_step={row['failed_step']}"
        )
    print(f"\nAll results saved under:\n  {RESULTS_DIR}")
    return 0 if all(result["status"] == "COMPLETED" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
