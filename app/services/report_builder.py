from typing import List, Optional
import random
import hashlib

from app.schemas import AnalysisReport, ModelSignal, TimelinePoint


def _risk_label(score: float) -> str:
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"


def _seed_from_key(seed_key: str) -> int:
    digest = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_report(
    report_id: str,
    filename: str,
    fps: float,
    frames_analyzed: int,
    approx_duration_sec: float,
    frame_fake_prob: float,
    frame_model_note: Optional[str] = None,
    frame_consistency: float = 1.0,
) -> AnalysisReport:
    """
    Hybrid report:
      - frame_artifacts_cnn uses REAL pretrained detector
      - temporal_consistency + audio_visual_sync remain placeholders for now
      - confidence now incorporates frame_consistency
    """
    rng = random.Random(_seed_from_key(report_id))

    cnn_score = float(min(1.0, max(0.0, frame_fake_prob)))

    # placeholder signals anchored near cnn_score
    temporal_score = min(1.0, max(0.0, cnn_score + rng.uniform(-0.15, 0.15)))
    avsync_score = min(1.0, max(0.0, cnn_score + rng.uniform(-0.20, 0.20)))

    signals: List[ModelSignal] = [
        ModelSignal(
            name="frame_artifacts_cnn",
            score=cnn_score,
            note=frame_model_note or "Frame detector (pretrained)",
        ),
        ModelSignal(
            name="temporal_consistency",
            score=temporal_score,
            note="Blink/micro-expression consistency (placeholder)",
        ),
        ModelSignal(
            name="audio_visual_sync",
            score=avsync_score,
            note="Lip-phoneme alignment check (placeholder)",
        ),
    ]

    manipulation_probability = sum(s.score for s in signals) / len(signals)

    # agreement among the 3 signals
    scores = [s.score for s in signals]
    spread = max(scores) - min(scores)
    model_agreement = 1.0 - spread

    # incorporate frame consistency (0..1) into confidence
    fc = float(min(1.0, max(0.0, frame_consistency)))

    # Confidence heuristic:
    # - rewards agreement across signals
    # - rewards stable frame-level behavior
    # - stays within [0.2, 0.95]
    confidence = 0.20 + 0.45 * model_agreement + 0.30 * fc
    confidence = max(0.20, min(0.95, confidence))

    trust_score = int(round((1.0 - manipulation_probability) * 100))
    editorial_risk = _risk_label(manipulation_probability)

    if editorial_risk == "high":
        recommendation = "Do not publish without independent verification; escalate to manual review."
    elif editorial_risk == "medium":
        recommendation = "Proceed with caution; verify source and cross-check with independent evidence."
    else:
        recommendation = "Low manipulation signals detected; continue standard editorial checks."

    # timeline (deterministic heuristic)
    timeline: List[TimelinePoint] = []
    segments = 6
    seg_len = approx_duration_sec / segments if approx_duration_sec > 0 else 2.0
    top_signals = ["audio_visual_sync", "temporal_consistency", "frame_artifacts_cnn"]

    base = manipulation_probability
    for i in range(segments):
        t0 = i * seg_len
        t1 = (i + 1) * seg_len
        seg_score = min(1.0, max(0.0, base + rng.uniform(-0.12, 0.12)))
        timeline.append(
            TimelinePoint(
                t_start_sec=round(t0, 2),
                t_end_sec=round(t1, 2),
                risk_score=round(seg_score, 3),
                risk_label=_risk_label(seg_score),
                top_signal=rng.choice(top_signals),
            )
        )

    return AnalysisReport(
        video_id=report_id,
        filename=filename,
        manipulation_probability=round(manipulation_probability, 3),
        trust_score=trust_score,
        confidence=round(confidence, 3),
        model_agreement=round(model_agreement, 3),
        editorial_risk=editorial_risk,  # type: ignore
        recommendation=recommendation,
        signals=signals,
        timeline=timeline,
        frames_analyzed=frames_analyzed,
        fps=round(float(fps), 3),
    )


def estimate_duration_sec(fps: float, num_frames_saved: int, every_n_frames: int) -> float:
    if fps <= 0:
        return 0.0
    approx_total_frames = num_frames_saved * max(1, every_n_frames)
    return approx_total_frames / fps