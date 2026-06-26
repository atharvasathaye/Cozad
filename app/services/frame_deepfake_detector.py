from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import os
import math
import statistics

from PIL import Image
from transformers import pipeline

MODEL_ID = os.environ.get("VERISTREAM_HF_MODEL", "prithivMLmods/Deep-Fake-Detector-v2-Model")

_PIPE: Optional[object] = None


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = pipeline(
            task="image-classification",
            model=MODEL_ID,
            device=-1,  # CPU
        )
    return _PIPE


def _fake_probability_from_predictions(preds: List[Dict[str, Any]]) -> float:
    """
    Convert predictions (top_k list) into a fake probability.

    This model uses labels like:
      - "Deepfake" (fake)
      - "Realism" (real)

    Strategy:
      - If we find an explicit fake label, use its score.
      - Else if we find an explicit real label, use 1 - real_score.
      - Else fallback to top-1 heuristic.
    """
    fake_score = None
    real_score = None

    for p in preds:
        label = str(p.get("label", "")).strip().lower()
        score = float(p.get("score", 0.0))

        if "fake" in label or "deepfake" in label:
            fake_score = score
        if "real" in label or "realism" in label:
            real_score = score

    if fake_score is not None:
        return float(fake_score)
    if real_score is not None:
        return float(1.0 - real_score)

    # fallback: interpret top prediction label
    if preds:
        top = preds[0]
        label = str(top.get("label", "")).strip().lower()
        score = float(top.get("score", 0.0))
        if "real" in label or "realism" in label:
            return float(1.0 - score)
        if "fake" in label or "deepfake" in label:
            return float(score)

    return 0.5


def _trimmed_mean(values: List[float], trim_frac: float = 0.2) -> float:
    """
    Compute trimmed mean by dropping trim_frac from each tail.
    Example: trim_frac=0.2 drops lowest 20% and highest 20%.
    """
    if not values:
        return 0.5
    if len(values) < 5:
        return float(sum(values) / len(values))

    v = sorted(values)
    k = int(math.floor(len(v) * trim_frac))
    mid = v[k: len(v) - k] if (len(v) - 2 * k) > 0 else v
    return float(sum(mid) / len(mid))


def score_frames_fake_probability(
    frame_paths: List[str],
    max_frames_to_score: int = 12,
    top_k: int = 5,
) -> Tuple[float, float, str]:
    """
    Scores a subset of frames with a pretrained deepfake image detector.

    Returns:
        avg_fake_probability: robust (trimmed mean) fake probability over sampled frames
        consistency: 1 - stdev (clipped) => higher means more consistent across frames
        note: human-readable note for report transparency
    """
    if not frame_paths:
        return 0.5, 0.0, "No frames provided; defaulted to 0.5"

    pipe = _get_pipe()

    # Sample evenly across the video
    if len(frame_paths) > max_frames_to_score:
        idxs = [round(i * (len(frame_paths) - 1) / (max_frames_to_score - 1)) for i in range(max_frames_to_score)]
        sampled = [frame_paths[i] for i in idxs]
    else:
        sampled = frame_paths

    probs: List[float] = []
    for p in sampled:
        img = Image.open(p).convert("RGB")
        preds = pipe(img, top_k=top_k)

        if isinstance(preds, list) and len(preds) > 0:
            probs.append(_fake_probability_from_predictions(preds))
        else:
            probs.append(0.5)

    # Robust aggregation: trimmed mean reduces impact of outliers
    avg = _trimmed_mean(probs, trim_frac=0.2)

    # Consistency: based on standard deviation across frame scores
    if len(probs) >= 2:
        stdev = statistics.pstdev(probs)
    else:
        stdev = 0.0

    # Map stdev (0..~0.5) into a "consistency" score (1=very consistent)
    # Clip stdev at 0.5 for safety.
    stdev_clipped = min(0.5, float(stdev))
    consistency = 1.0 - (stdev_clipped / 0.5)

    note = (
        f"Scored {len(sampled)} frames using {MODEL_ID}; "
        f"agg=trimmed_mean(20%); min={min(probs):.3f}, max={max(probs):.3f}, stdev={stdev:.3f}"
    )
    return float(avg), float(consistency), note


def debug_predict_one_frame(frame_path: str, top_k: int = 5) -> Dict[str, Any]:
    pipe = _get_pipe()
    img = Image.open(frame_path).convert("RGB")
    out = pipe(img, top_k=top_k)
    return {
        "model_id": MODEL_ID,
        "frame_path": frame_path,
        "predictions": out,
    }