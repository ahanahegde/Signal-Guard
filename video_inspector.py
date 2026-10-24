
import cv2
from pathlib import Path


def inspect_video(video_path):
    path = Path(video_path)

    if not path.exists():
        return {"error": "Video file not found."}

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return {"error": "Could not open video."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0 or frames <= 0:
        return {"error": "Invalid video metadata."}

    return {
        "filename": path.name,
        "duration_seconds": round(frames / fps, 2),
        "fps": round(fps, 2),
        "resolution": f"{width}x{height}",
        "frame_count": int(frames),
    }


if __name__ == "__main__":
    video = input("Enter video path: ").strip().strip('"')
    result = inspect_video(video)

    for key, value in result.items():
        print(f"{key}: {value}")
