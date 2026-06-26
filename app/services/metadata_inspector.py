from __future__ import annotations

import json
import os
import subprocess
import shutil
from typing import Dict, Any, List


def _resolve_ffprobe() -> str:
    """
    Resolve ffprobe executable path reliably on Windows.

    Priority:
      1) FFPROBE_PATH env var (absolute path to ffprobe.exe)
      2) ffprobe found on PATH via shutil.which
      3) common install locations (best-effort)
    """
    env_path = os.environ.get("FFPROBE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    found = shutil.which("ffprobe")
    if found:
        return found

    # Best-effort common locations (Gyan dev build often ends up here if user chose it)
    common = [
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe",
    ]
    for p in common:
        if os.path.exists(p):
            return p

    raise FileNotFoundError("ffprobe not found. Set FFPROBE_PATH or add ffprobe to PATH.")


def _run_ffprobe(video_path: str) -> Dict[str, Any]:
    ffprobe = _resolve_ffprobe()

    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise RuntimeError(err if err else "ffprobe failed with non-zero exit code")

    return json.loads(proc.stdout)


def inspect_metadata(video_path: str) -> Dict[str, Any]:
    data = _run_ffprobe(video_path)

    format_info = data.get("format", {})
    streams = data.get("streams", [])

    tags = format_info.get("tags", {}) or {}
    warnings: List[str] = []

    encoder = (
        tags.get("encoder")
        or tags.get("ENCODER")
        or tags.get("software")
        or tags.get("SOFTWARE")
    )
    if not encoder:
        warnings.append("encoder_missing_or_stripped")

    creation_time = tags.get("creation_time") or tags.get("CREATION_TIME")

    make = tags.get("com.apple.quicktime.make") or tags.get("make") or tags.get("MAKE")
    model = tags.get("com.apple.quicktime.model") or tags.get("model") or tags.get("MODEL")

    device_confidence = "low"
    if make and model:
        device_confidence = "medium"

    container = {
        "format": format_info.get("format_name"),
        "major_brand": tags.get("major_brand"),
        "minor_version": tags.get("minor_version"),
    }

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    stream_summary = []
    for s in video_streams:
        stream_summary.append(
            {
                "codec": s.get("codec_name"),
                "width": s.get("width"),
                "height": s.get("height"),
                "fps": s.get("avg_frame_rate"),
                "bit_rate": s.get("bit_rate"),
                "color_space": s.get("color_space"),
            }
        )

    if not make and not model:
        warnings.append("device_metadata_missing_or_stripped")

    return {
        "device": {"make": make, "model": model, "confidence": device_confidence},
        "encoder": encoder,
        "creation_time": creation_time,
        "container": container,
        "streams": stream_summary,
        "warnings": warnings,
    }
