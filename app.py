import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import os

from modules.proxy      import ProxyDetector
from modules.hand_raise import HandRaiseDetector
from modules.stress     import StressDetector
from modules.engagement import EngagementDetector

# =====================================================
#   VIRTUAL CLASSROOM ATTENTION MONITORING SYSTEM
# =====================================================

STRESS_MODEL_PATH     = os.path.join("models", "stress_model_da.h5")
ENGAGEMENT_MODEL_PATH = os.path.join("models", "engagement_model_best.h5")
ML_FRAME_SKIP         = 5

# =====================================================
#   PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Virtual Classroom Monitor", layout="wide")
st.title("Virtual Classroom Attention Monitoring System")
st.markdown("---")

# =====================================================
#   LOAD MODELS — cached, loaded once only
# =====================================================

@st.cache_resource
def load_detectors():
    stress     = StressDetector(STRESS_MODEL_PATH)
    engagement = EngagementDetector(ENGAGEMENT_MODEL_PATH)
    proxy      = ProxyDetector()
    hand_raise = HandRaiseDetector()
    return stress, engagement, proxy, hand_raise

@st.cache_resource
def load_mediapipe():
    pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    return pose, face_detection, face_mesh

# =====================================================
#   SESSION STATE — prevents re-run crash
# =====================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "stress_result" not in st.session_state:
    st.session_state.stress_result     = {"label": "Detecting...", "confidence": 0}
if "engagement_result" not in st.session_state:
    st.session_state.engagement_result = {"label": "Detecting...", "confidence": 0,
                                           "source": "—", "mp_details": {}}
if "handrise_result" not in st.session_state:
    st.session_state.handrise_result   = {"hand_raised": False,
                                           "raise_type": None, "hand_side": None}
if "proxy_result" not in st.session_state:
    st.session_state.proxy_result      = {
        "proxy_detected": False, "proxy_score": 0,
        "multiple_faces": False, "face_absent": False,
        "static_face": False, "looking_away": False,
        "gaze_detail": ""
    }

# =====================================================
#   SIDEBAR
# =====================================================

st.sidebar.title("Controls")

# Toggle button — uses session_state so sidebar clicks don't kill the loop
if st.sidebar.button(
    "Stop Monitoring" if st.session_state.running else "Start Monitoring"
):
    st.session_state.running = not st.session_state.running

st.sidebar.markdown("---")
st.sidebar.markdown("### Modules")
stress_enabled     = st.sidebar.checkbox("Stress Detection",     value=True)
engagement_enabled = st.sidebar.checkbox("Engagement Analysis",  value=True)
handrise_enabled   = st.sidebar.checkbox("Hand Raise Detection", value=True)
proxy_enabled      = st.sidebar.checkbox("Proxy Detection",      value=True)

st.sidebar.markdown("---")
st.sidebar.info("ML models run every 5th frame. MediaPipe runs every frame.")

# =====================================================
#   LAYOUT
# =====================================================

col_video, col_status = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()
    status_text       = st.empty()

with col_status:
    st.markdown("### Live Status")
    stress_box     = st.empty()
    engagement_box = st.empty()
    handrise_box   = st.empty()
    proxy_box      = st.empty()
    proxy_flags    = st.empty()

# =====================================================
#   CARD HELPERS
# =====================================================

def status_card(label, value, alert=False, neutral=False):
    if neutral:
        color = "#888888"
    else:
        color = "#ff4444" if alert else "#22cc44"
    return f"""
    <div style="background:#1e1e1e; border-left:4px solid {color};
                border-radius:6px; padding:10px 14px; margin-bottom:10px;">
        <div style="color:#aaa; font-size:12px;">{label}</div>
        <div style="color:{color}; font-size:16px; font-weight:bold;">{value}</div>
    </div>"""

def proxy_flag_card(flags_dict, conditional_flags=None):
    """
    flags_dict         : always-shown flags {name: triggered}
    conditional_flags  : only-shown-when-triggered flags {name: triggered}
    """
    rows = ""
    # Always-on flags — show green or red
    for name, triggered in flags_dict.items():
        color  = "#ff4444" if triggered else "#22cc44"
        status = "⚠ DETECTED" if triggered else "✓ OK"
        rows  += f"""
        <div style="display:flex; justify-content:space-between;
                    padding:4px 0; border-bottom:1px solid #333;">
            <span style="color:#ccc; font-size:13px;">{name}</span>
            <span style="color:{color}; font-size:13px; font-weight:bold;">{status}</span>
        </div>"""
    # Conditional flags — only shown when triggered
    if conditional_flags:
        for name, triggered in conditional_flags.items():
            if triggered:
                rows += f"""
        <div style="display:flex; justify-content:space-between;
                    padding:4px 0; border-bottom:1px solid #333;">
            <span style="color:#ccc; font-size:13px;">{name}</span>
            <span style="color:#ff4444; font-size:13px; font-weight:bold;">⚠ DETECTED</span>
        </div>"""
    return f"""
    <div style="background:#1e1e1e; border-radius:6px;
                padding:10px 14px; margin-bottom:10px;">
        <div style="color:#aaa; font-size:12px; margin-bottom:6px;">Proxy Flags</div>
        {rows}
    </div>"""

def render_sidebar(sr, er, hr, pr):
    """Render all status cards from latest results."""

    # Stress
    stress_box.markdown(
        status_card(
            "Stress",
            f"{sr['label']} ({sr['confidence']}%)",
            alert   = sr["label"] == "STRESS",
            neutral = sr["label"] == "Detecting..."
        ), unsafe_allow_html=True
    )

    # Engagement
    eng_label = er["label"]
    engagement_box.markdown(
        status_card(
            "Engagement",
            f"{eng_label} · {er['source']}",
            alert   = eng_label in ["Low", "Very Low"],
            neutral = eng_label == "Detecting..."
        ), unsafe_allow_html=True
    )

    # Hand Raise
    if hr["hand_raised"]:
        hr_value = f"YES — {hr['raise_type']} ({hr['hand_side']})"
        hr_alert = True
    else:
        hr_value = "NO"
        hr_alert = False
    handrise_box.markdown(
        status_card("Hand Raised", hr_value, alert=hr_alert),
        unsafe_allow_html=True
    )

    # Proxy summary
    proxy_status = "⚠ PROXY DETECTED" if pr["proxy_detected"] else "Normal"
    proxy_box.markdown(
        status_card("Proxy", proxy_status, alert=pr["proxy_detected"]),
        unsafe_allow_html=True
    )

    # Gaze label
    gaze_label = "Looking Away"
    if pr.get("gaze_detail"):
        gaze_label = f"Looking Away ({pr['gaze_detail']})"

    proxy_flags.markdown(
        proxy_flag_card(
            flags_dict = {
                "Multiple Faces": pr["multiple_faces"],
                "Static Face"   : pr["static_face"],
                gaze_label      : pr["looking_away"],
            },
            conditional_flags = {
                "Face Absent": pr["face_absent"],
            }
        ), unsafe_allow_html=True
    )

# =====================================================
#   RENDER WAITING STATE
# =====================================================

def render_waiting():
    for box in [stress_box, engagement_box, handrise_box, proxy_box]:
        box.markdown(status_card("—", "Waiting...", neutral=True),
                     unsafe_allow_html=True)
    proxy_flags.markdown(
        proxy_flag_card(
            flags_dict = {
                "Multiple Faces": False,
                "Static Face"   : False,
                "Looking Away"  : False,
            }
        ), unsafe_allow_html=True
    )

# =====================================================
#   MAIN LOOP
# =====================================================

if st.session_state.running:
    stress_det, engage_det, proxy_det, hand_det = load_detectors()
    pose, face_detection, face_mesh             = load_mediapipe()

    cap         = cv2.VideoCapture(0)
    frame_count = 0

    if not cap.isOpened():
        st.error("Cannot access webcam. Check if another app is using it.")
        st.session_state.running = False
        st.stop()

    status_text.info("Monitoring active — click Stop Monitoring to end.")

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam read failed.")
            break

        frame        = cv2.flip(frame, 1)
        h, w, _      = frame.shape
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1

        # ---- MediaPipe — every frame ----
        pose_results = pose.process(rgb)
        det_results  = face_detection.process(rgb)
        mesh_results = face_mesh.process(rgb)

        # ---- Hand Raise — every frame ----
        if handrise_enabled:
            st.session_state.handrise_result = hand_det.update(pose_results, h)

        # ---- Proxy — every frame ----
        if proxy_enabled:
            st.session_state.proxy_result = proxy_det.update(det_results, mesh_results)

        # ---- ML Models — every N frames ----
        if frame_count % ML_FRAME_SKIP == 0:
            if stress_enabled:
                st.session_state.stress_result = stress_det.update(frame, mesh_results)
            if engagement_enabled:
                st.session_state.engagement_result = engage_det.update(frame, mesh_results)

        # ---- Draw overlays on frame ----
        display = frame.copy()
        if handrise_enabled:
            display = hand_det.draw(display, st.session_state.handrise_result)
        if proxy_enabled:
            display = proxy_det.draw(display, st.session_state.proxy_result)

        # ---- Show frame ----
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        video_placeholder.image(display_rgb, channels="RGB", use_column_width=True)

        # ---- Update status cards ----
        render_sidebar(
            st.session_state.stress_result,
            st.session_state.engagement_result,
            st.session_state.handrise_result,
            st.session_state.proxy_result
        )

    cap.release()
    status_text.info("Monitoring stopped.")

else:
    video_placeholder.info("Click 'Start Monitoring' in the sidebar to begin.")
    render_waiting()