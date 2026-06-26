from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
import hashlib

from app.services.video_frames import extract_frames
from app.services.report_builder import build_report, estimate_duration_sec
from app.services.frame_deepfake_detector import score_frames_fake_probability, debug_predict_one_frame
from app.services.report_store import save_report, load_report
from app.services.quality_gate import estimate_video_quality, quality_flag
from app.services.metadata_inspector import inspect_metadata

app = FastAPI(title="VeriStream AI", version="0.9.0")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

EVERY_N_FRAMES = 5
MAX_FRAMES = 60


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    file_bytes = await file.read()

    report_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    upload_id = str(uuid.uuid4())

    video_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{file.filename}")
    with open(video_path, "wb") as f:
        f.write(file_bytes)

    # --- Metadata / provenance inspection ---
    try:
        provenance = inspect_metadata(video_path)
    except Exception as e:
        provenance = {
            "error": str(e),
            "warnings": ["ffprobe_failed"],
        }

    frames_dir = os.path.join(UPLOAD_DIR, f"{upload_id}_frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)

    frame_paths, fps = extract_frames(
        video_path=video_path,
        out_dir=frames_dir,
        every_n_frames=EVERY_N_FRAMES,
        max_frames=MAX_FRAMES,
    )

    approx_duration = estimate_duration_sec(
        fps=fps,
        num_frames_saved=len(frame_paths),
        every_n_frames=EVERY_N_FRAMES,
    )

    # Quality gate
    q = estimate_video_quality(frame_paths, max_frames=12)
    q_status = quality_flag(blur_avg=q["blur_avg"], face_rate=q["face_rate"])

    # Deepfake detector
    frame_fake_prob, frame_consistency, frame_note = score_frames_fake_probability(
        frame_paths=frame_paths,
        max_frames_to_score=12,
        top_k=5,
    )

    report = build_report(
        report_id=report_id,
        filename=file.filename,
        fps=fps,
        frames_analyzed=len(frame_paths),
        approx_duration_sec=approx_duration,
        frame_fake_prob=frame_fake_prob,
        frame_model_note=frame_note,
        frame_consistency=frame_consistency,
    )

    out = report.model_dump()
    out["report_id"] = report_id
    out["upload_id"] = upload_id
    out["frames_dir"] = frames_dir
    out["frame_consistency"] = round(float(frame_consistency), 3)

    out["quality"] = {
        "status": q_status,
        "blur_avg": round(float(q["blur_avg"]), 3),
        "face_rate": round(float(q["face_rate"]), 3),
        "frames_checked": int(q["frames_checked"]),
    }

    out["provenance"] = provenance

    if q_status != "ok":
        out["confidence"] = min(float(out.get("confidence", 0.5)), 0.45)
        out["recommendation"] = (
            "Insufficient forensic quality. Automated assessment may be unreliable; "
            "verify original source and metadata."
        )

    save_report(report_id=report_id, report=out)
    return out


@app.get("/report/{report_id}")
def get_report(report_id: str):
    try:
        return load_report(report_id=report_id)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Report not found"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/debug/frame-predict")
def debug_frame_predict(
    frame_path: str = Query(...),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        return debug_predict_one_frame(frame_path=frame_path, top_k=top_k)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})