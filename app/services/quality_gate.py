from __future__ import annotations

from typing import List, Dict
import cv2
import numpy as np


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_video_quality(frame_paths: List[str], max_frames: int = 12) -> Dict[str, float]:
    """
    Simple forensic quality checks:
      - blur score (Laplacian variance): lower => blurrier
      - face present rate using Haar cascade (fast, coarse)
    """
    if not frame_paths:
        return {"blur_avg": 0.0, "face_rate": 0.0, "frames_checked": 0.0}

    # sample evenly
    if len(frame_paths) > max_frames:
        idxs = [round(i * (len(frame_paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
        sampled = [frame_paths[i] for i in idxs]
    else:
        sampled = frame_paths

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    blur_vals = []
    face_hits = 0

    for p in sampled:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur_vals.append(variance_of_laplacian(gray))

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            face_hits += 1

    frames_checked = float(len(blur_vals))
    if frames_checked == 0:
        return {"blur_avg": 0.0, "face_rate": 0.0, "frames_checked": 0.0}

    blur_avg = float(sum(blur_vals) / len(blur_vals))
    face_rate = float(face_hits / len(blur_vals))

    return {"blur_avg": blur_avg, "face_rate": face_rate, "frames_checked": frames_checked}


def quality_flag(blur_avg: float, face_rate: float) -> str:
    """
    Conservative thresholds for demo:
      - blur_avg < ~60 often indicates significant blur/compression
      - face_rate < 0.5 means detector rarely sees a face (not ideal for face-forensics)
    """
    if face_rate < 0.5:
        return "low_face_coverage"
    if blur_avg < 60.0:
        return "high_blur"
    return "ok"