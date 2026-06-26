import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="VeriStream AI Demo", layout="wide")

API_BASE = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")

st.title("VeriStream AI")
st.caption("Manipulation risk scoring for short-form video (prototype).")

uploaded = st.file_uploader("Upload a video file (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
analyze = st.button("Analyze", type="primary", disabled=(uploaded is None))

def stability_label(frame_consistency: float) -> str:
    if frame_consistency >= 0.75:
        return "Stable"
    if frame_consistency >= 0.50:
        return "Mixed"
    return "Unstable"

if analyze:
    with st.spinner("Uploading and analyzing..."):
        try:
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "video/mp4")}
            resp = requests.post(f"{API_BASE}/analyze/video", files=files, timeout=300)

            if resp.status_code != 200:
                st.error(f"API error ({resp.status_code}): {resp.text}")
                st.stop()

            report = resp.json()

        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API at {API_BASE}. Error: {e}")
            st.stop()

    trust_score = report.get("trust_score", None)
    manip_prob = float(report.get("manipulation_probability", 0.0))
    confidence = float(report.get("confidence", 0.0))
    agreement = float(report.get("model_agreement", 0.0))
    frame_consistency = float(report.get("frame_consistency", 0.0))

    # --- Top KPIs ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trust Score", trust_score if trust_score is not None else "—")
    c2.metric("Manipulation Probability", f"{manip_prob:.3f}")
    c3.metric("Confidence", f"{confidence:.3f}")
    c4.metric("Model Agreement", f"{agreement:.3f}")
    c5.metric("Frame Consistency", f"{frame_consistency:.3f}")

    # --- Stability banner ---
    st.subheader("Stability")
    label = stability_label(frame_consistency)
    if label == "Stable":
        st.success("Stable signal across frames. Results are more reliable.")
    elif label == "Mixed":
        st.warning("Mixed signal across frames. Consider manual review of flagged segments.")
    else:
        st.error("Unstable signal across frames. High chance of noise (compression, motion blur). Manual review recommended.")

    # --- Editorial ---
    st.subheader("Editorial Summary")
    st.write(f"**Risk:** {str(report.get('editorial_risk', '—')).upper()}")
    st.write(f"**Recommendation:** {report.get('recommendation', '—')}")

    # --- Signals ---
    st.subheader("Signals")
    signals = report.get("signals", [])
    if signals:
        sig_df = pd.DataFrame(signals)
        if "score" in sig_df.columns:
            sig_df["score"] = sig_df["score"].astype(float)

        st.dataframe(sig_df, use_container_width=True)
        if "name" in sig_df.columns and "score" in sig_df.columns:
            st.bar_chart(sig_df.set_index("name")["score"])
    else:
        st.info("No signals returned.")

    # --- Timeline ---
    st.subheader("Risk Timeline")
    timeline = report.get("timeline", [])
    if timeline:
        tl_df = pd.DataFrame(timeline)
        tl_df["risk_score"] = tl_df["risk_score"].astype(float)

        st.dataframe(tl_df, use_container_width=True)

        chart_df = tl_df[["t_start_sec", "risk_score"]].rename(columns={"t_start_sec": "time_sec"}).set_index("time_sec")
        st.line_chart(chart_df)

        max_row = tl_df.iloc[tl_df["risk_score"].idxmax()]
        st.warning(
            f"Highest risk segment: {max_row['t_start_sec']}s–{max_row['t_end_sec']}s "
            f"(score={max_row['risk_score']}, top_signal={max_row['top_signal']})"
        )
    else:
        st.info("No timeline returned.")

    # --- Raw JSON ---
    with st.expander("Raw API Response (JSON)"):
        st.json(report)

else:
    st.info("Start your FastAPI server, then upload a video and click Analyze.")
    st.code("python -m uvicorn app.main:app --reload", language="bash")