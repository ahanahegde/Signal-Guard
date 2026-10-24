
import cv2
import json
import argparse
import tempfile
from pathlib import Path


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def transform_video(source, output, mode):
    cap = open_video(source)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata.")

    # MP4 output for all transformed copies.
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output: {output}")

    frame_index = 0
    written_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1

        if mode == "temporal":
            # Keep approximately 90% of frames.
            # Output FPS is adjusted to preserve approximate duration.
            if frame_index % 10 == 0:
                continue

        elif mode == "bright":
            # Mild brightness increase.
            frame = cv2.convertScaleAbs(
                frame, alpha=1.0, beta=12
            )

        elif mode == "dark":
            # Mild brightness decrease.
            frame = cv2.convertScaleAbs(
                frame, alpha=1.0, beta=-12
            )

        elif mode == "compression":
            # Decode and re-encode with additional JPEG artifacts.
            # This stresses frame appearance; MP4 is then written below.
            frame = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 45]
            )[1]
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        writer.write(frame)
        written_frames += 1

    cap.release()
    writer.release()

    if written_frames == 0:
        raise RuntimeError(f"No frames written for {mode}")

    if mode == "temporal":
        # Preserve approximate playback duration after dropping frames.
        # Re-encode the output at 90% of the original frame rate.
        adjusted = output.with_name(output.stem + "_adjusted.mp4")
        source_cap = cv2.VideoCapture(str(output))
        new_writer = cv2.VideoWriter(
            str(adjusted),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps * 0.9,
            (width, height)
        )

        while True:
            ok, frame = source_cap.read()
            if not ok:
                break
            new_writer.write(frame)

        source_cap.release()
        new_writer.release()
        output.unlink()
        adjusted.rename(output)

    return {
        "file": str(output),
        "frames_written": written_frames,
        "original_fps": round(fps, 3),
        "test": mode
    }


def main():
    parser = argparse.ArgumentParser(
        description="SignalGuard video stress-test variant generator"
    )
    parser.add_argument("video", help="Path to the original video")
    parser.add_argument(
        "--out",
        default="stress_outputs",
        help="Output folder (default: stress_outputs)"
    )
    args = parser.parse_args()

    source = Path(args.video).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = [
        ("compression", "compression.mp4"),
        ("bright", "brightness_up.mp4"),
        ("dark", "brightness_down.mp4"),
        ("temporal", "temporal_drop.mp4"),
    ]

    results = []
    for mode, filename in tests:
        destination = out_dir / filename
        try:
            result = transform_video(source, destination, mode)
            results.append(result)
            print(f"[CREATED] {mode}: {destination}")
        except Exception as exc:
            results.append({
                "test": mode,
                "error": str(exc)
            })
            print(f"[FAILED] {mode}: {exc}")

    manifest = out_dir / "stress_manifest.json"
    manifest.write_text(
        json.dumps({
            "original_video": str(source),
            "warning": (
                "Generated variants only. No deepfake verdict is made. "
                "Compare physiological metrics using the SignalGuard "
                "analysis pipeline."
            ),
            "results": results
        }, indent=2),
        encoding="utf-8"
    )

    print(f"\nManifest saved: {manifest}")
    print("Original video was not modified.")
    print("Next: run the POS and eligibility pipeline on each variant.")


if __name__ == "__main__":
    main()
