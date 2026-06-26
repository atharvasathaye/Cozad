import os
from typing import List, Tuple

import cv2


def extract_frames(
    video_path: str,
    out_dir: str,
    every_n_frames: int = 5,
    max_frames: int = 60,
) -> Tuple[List[str], float]:
    """
    Extract frames from a video file.

    Args:
        video_path: path to the video on disk
        out_dir: directory where frames will be saved
        every_n_frames: sample every Nth frame
        max_frames: maximum number of frames to save

    Returns:
        (frame_paths, fps)
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    frame_paths: List[str] = []
    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n_frames == 0:
            frame_file = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_file, frame)
            frame_paths.append(frame_file)
            saved += 1

            if saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frame_paths, float(fps)