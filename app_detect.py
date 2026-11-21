"""EcoShot AI multi-object detection page (separate from the E2 classification demo app.py).

Run from the repository root:
    python -m streamlit run app_detect.py

It needs a TRAINED detector at Models/e6/best.pt with Models/e6/protocol.json (override the
folder with the ECOSHOT_E6_DIR environment variable). Without them the page says so and
shows nothing: it never draws made-up boxes.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import streamlit as st

from Src.detector.app_logic import (
    DetectorError,
    UploadError,
    build_report,
    run_detection,
    validate_upload,
)
from Src.detector.classes import GUIDANCE_NOTICE
from Src.detector.e6_detector import E6Detector, load_rgb_image

MODEL_DIR = Path(os.environ.get("ECOSHOT_E6_DIR", "Models/e6"))

st.set_page_config(page_title="EcoShot AI - Waste Detection", layout="wide")


@st.cache_resource(show_spinner="Loading detector ...")
def load_detector(model_dir: str) -> E6Detector:
    base = Path(model_dir)
    return E6Detector(base / "best.pt", base / "protocol.json")


st.title("EcoShot AI: multi-object waste detection")
st.caption(
    "Finds each visible waste object, draws its own box and gives its category. "
    "Detection scores are uncalibrated, and the model cannot say \"unknown object\"."
)

try:
    detector = load_detector(str(MODEL_DIR))
except DetectorError as exc:
    st.error(f"The detector is not ready: {exc}")
    st.info(
        f"Expected a trained checkpoint at `{MODEL_DIR / 'best.pt'}` and its protocol at "
        f"`{MODEL_DIR / 'protocol.json'}`. Train the E6 detector first (see Documentation/E6_DETECTOR_WORKFLOW.md). "
        "This page will not show any boxes until a real checkpoint is loaded."
    )
    st.stop()

with st.sidebar:
    st.subheader("Model")
    st.write(f"Checkpoint SHA-256: `{detector.checkpoint_sha256[:12]}...`")
    st.write(f"Input size: {detector.imgsz} px")
    default_conf = float(detector.conf_threshold)
    conf = st.slider(
        "Detection threshold",
        min_value=min(0.05, default_conf),  # bounds always contain the validated default
        max_value=max(0.90, default_conf),
        value=default_conf,
        step=0.01,
        format="%.2f",
        help="Lower finds more objects but adds false detections. The default comes from validation, not from you.",
    )
    st.caption("Scores are raw detector scores, not probabilities of being correct.")

uploaded = st.file_uploader("Upload a photo (JPEG or PNG, up to 10 MB)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.session_state.pop("outcome", None)
    st.info("Upload a photo that contains one or more waste objects.")
    st.stop()

data = uploaded.getvalue()
try:
    validate_upload(uploaded.name, data)
except UploadError as exc:
    st.session_state.pop("outcome", None)
    st.error(str(exc))
    st.stop()

key = f"{uploaded.name}:{hashlib.sha256(data).hexdigest()}:{conf:.2f}"
stored = st.session_state.get("outcome")
if stored is not None and stored[0] != key:
    st.session_state.pop("outcome", None)  # never show results for a different image or threshold
    stored = None

left, right = st.columns(2)
with left:
    st.subheader("Original")
    try:
        st.image(load_rgb_image(io.BytesIO(data)), use_container_width=True)
    except DetectorError as exc:
        st.error(str(exc))
        st.stop()

if st.button("Detect objects", type="primary"):
    try:
        with st.spinner("Detecting ..."):
            outcome = run_detection(detector, data, conf)
        st.session_state["outcome"] = (key, outcome)
        stored = (key, outcome)
    except DetectorError as exc:
        st.session_state.pop("outcome", None)
        st.error(str(exc))
        stored = None

if stored is not None:
    outcome = stored[1]
    with right:
        st.subheader("Detected")
        st.image(outcome.annotated_png, use_container_width=True)

    total = len(outcome.object_rows)
    if total == 0:
        st.warning(
            "No objects were detected above the threshold. That does not prove the photo has no waste: "
            "try a lower threshold, a closer photo or better lighting."
        )
    else:
        st.subheader(f"{total} object{'s' if total != 1 else ''} detected")
        st.dataframe(
            [{"Category": k, "Count": v} for k, v in outcome.counts.items() if v],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**Each object**")
        st.dataframe(outcome.object_rows, hide_index=True, use_container_width=True)
        st.markdown("**Disposal guidance**")
        for row in outcome.guidance_rows:
            st.markdown(f"- **{row['category'].capitalize()}** ({row['count']}): {row['advice']}")
        st.caption(GUIDANCE_NOTICE)
    if outcome.result.dropped_invalid:
        st.caption(f"{outcome.result.dropped_invalid} invalid model box(es) were discarded.")

    report = build_report(outcome, uploaded.name, data)
    d1, d2 = st.columns(2)
    d1.download_button("Download result JSON", json.dumps(report, indent=2), file_name="ecoshot_detection.json", mime="application/json")
    d2.download_button("Download annotated image", outcome.annotated_png, file_name="ecoshot_detection.png", mime="image/png")
