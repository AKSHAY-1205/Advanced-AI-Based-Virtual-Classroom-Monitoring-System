import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import os
import threading

from modules.proxy      import ProxyDetector
from modules.hand_raise import HandRaiseDetector
from modules.stress     import StressDetector
from modules.engagement import EngagementDetector

# =====================================================
#   VIRTUAL CLASSROOM ATTENTION MONITORING SYSTEM
# =====================================================

STRESS_MODEL_PATH     = os.path.join("models", "stress_model_da.h5")
ENGAGEMENT_MODEL_PATH = os.path.join("models", "engagement_model_best.h5")

# How often to update sidebar cards (every N frames)
# Video runs at full speed — sidebar only updates occasionally to reduce lag
SIDEBAR_UPDATE_EVERY  = 15

# =====================================================
#   LIGHTWEIGHT FRAME DRAWING — pure OpenCV, no PIL
#   PIL was creating/destroying image objects every frame causing lag
# =====================================================

def _draw_label(frame, text, x, y, color, scale=0.55, thickness=1):
    """Draw a label with dark background pill. Fast OpenCV only."""
    font = cv2.FONT_HERSHEY_DUPLEX   # cleaner than SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + pad),
        (15, 15, 15), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_frame_overlays(frame, stress_r, engage_r, hand_r, proxy_r, mesh_results):
    """
    Draw all detection labels + face bounding box on frame.
    Uses cv2.FONT_HERSHEY_DUPLEX with LINE_AA for clean antialiased text.
    No PIL — pure OpenCV for maximum speed.
    """

    h, w  = frame.shape[:2]
    lx    = 12
    y     = 28

    C_OK      = (40, 210, 40)     # green
    C_ALERT   = (50, 60, 240)     # red
    C_NEUTRAL = (190, 190, 190)   # grey
    C_ORANGE  = (30, 160, 255)    # orange

    # ---- Face bounding box from FaceMesh ----
    if mesh_results and mesh_results.multi_face_landmarks:
        lm  = mesh_results.multi_face_landmarks[0].landmark
        xs  = [l.x * w for l in lm]
        ys  = [l.y * h for l in lm]
        x1  = max(0,  int(min(xs)) - 12)
        y1  = max(0,  int(min(ys)) - 12)
        x2  = min(w,  int(max(xs)) + 12)
        y2  = min(h,  int(max(ys)) + 12)
        box_color = C_ALERT if stress_r["label"] == "STRESS" else C_OK
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # ---- Proxy always-on flags ----
    gaze_label = (f"Looking Away ({proxy_r['gaze_detail']})"
                  if proxy_r.get("gaze_detail") else "Looking Away")

    proxy_always = [
        ("Multiple Faces", proxy_r["multiple_faces"]),
        ("Static Face",    proxy_r["static_face"]),
        (gaze_label,       proxy_r["looking_away"]),
    ]
    for label, triggered in proxy_always:
        color  = C_ALERT if triggered else C_OK
        status = "DETECTED" if triggered else "OK"
        _draw_label(frame, f"{label}: {status}", lx, y, color)
        y += 26

    # Face absent — conditional
    if proxy_r["face_absent"]:
        _draw_label(frame, "Face Absent: DETECTED", lx, y, C_ALERT)
        y += 26

    # ---- Hand Raise ----
    if hand_r["hand_raised"]:
        txt   = f"Hand Raised: {hand_r['raise_type']} ({hand_r['hand_side']})"
        color = C_OK
    else:
        txt   = "Hand Raised: NO"
        color = C_OK
    _draw_label(frame, txt, lx, y, color)
    y += 26

    # ---- Stress ----
    s_label = stress_r["label"]
    s_conf  = stress_r["confidence"]
    s_color = (C_ALERT   if s_label == "STRESS"    else
               C_OK      if s_label == "NO STRESS" else C_NEUTRAL)
    _draw_label(frame, f"Stress: {s_label} ({s_conf}%)", lx, y, s_color, scale=0.60)
    y += 26

    # ---- Engagement ----
    e_label = engage_r["label"]
    e_color = (C_ALERT  if e_label in ["Very Low", "Low"]   else
               C_OK     if e_label in ["High", "Very High"] else C_NEUTRAL)
    _draw_label(frame, f"Engagement: {e_label}", lx, y, e_color)
    y += 26

    # ---- Proxy banner ----
    if proxy_r["proxy_detected"]:
        banner = "!! PROXY DETECTED !!"
        font   = cv2.FONT_HERSHEY_DUPLEX
        scale  = 0.75
        thick  = 2
        (bw, bh), _ = cv2.getTextSize(banner, font, scale, thick)
        bx = lx
        by = y + 8
        cv2.rectangle(frame, (bx-6, by-bh-6), (bx+bw+6, by+6), (0, 0, 160), -1)
        cv2.putText(frame, banner, (bx, by), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    return frame


# =====================================================
#   PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Virtual Classroom Monitor", layout="wide")
st.title("Virtual Classroom Attention Monitoring System")
st.markdown("---")

# =====================================================
#   LOAD MODELS — cached once
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
#   SESSION STATE
# =====================================================

if "running" not in st.session_state:
    st.session_state.running = False
if "stress_result" not in st.session_state:
    st.session_state.stress_result     = {"label": "Detecting...", "confidence": 0}
if "engagement_result" not in st.session_state:
    st.session_state.engagement_result = {"label": "Detecting...",
                                           "confidence": 0, "source": "—", "mp_details": {}}
if "handrise_result" not in st.session_state:
    st.session_state.handrise_result   = {"hand_raised": False,
                                           "raise_type": None, "hand_side": None}
if "proxy_result" not in st.session_state:
    st.session_state.proxy_result      = {
        "proxy_detected": False, "proxy_score": 0,
        "multiple_faces": False, "face_absent": False,
        "static_face": False, "looking_away": False, "gaze_detail": ""
    }

# =====================================================
#   SIDEBAR
# =====================================================

st.sidebar.title("Controls")
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
st.sidebar.info("Inference runs in background thread for smooth video.")

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
#   SIDEBAR CARD HELPERS
# =====================================================

def status_card(label, value, alert=False, neutral=False):
    color = "#888888" if neutral else ("#ff4444" if alert else "#22cc44")
    return f"""
    <div style="background:#1e1e1e; border-left:4px solid {color};
                border-radius:6px; padding:10px 14px; margin-bottom:10px;">
        <div style="color:#aaa; font-size:12px;">{label}</div>
        <div style="color:{color}; font-size:16px; font-weight:bold;">{value}</div>
    </div>"""

def proxy_flag_card(always_flags, conditional_flags=None):
    rows = ""
    for name, triggered in always_flags.items():
        color  = "#ff4444" if triggered else "#22cc44"
        status = "⚠ DETECTED" if triggered else "✓ OK"
        rows  += f"""
        <div style="display:flex; justify-content:space-between;
                    padding:4px 0; border-bottom:1px solid #333;">
            <span style="color:#ccc; font-size:13px;">{name}</span>
            <span style="color:{color}; font-size:13px; font-weight:bold;">{status}</span>
        </div>"""
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
    stress_box.markdown(
        status_card("Stress",
            f"{sr['label']} ({sr['confidence']}%)",
            alert   = sr["label"] == "STRESS",
            neutral = sr["label"] == "Detecting..."),
        unsafe_allow_html=True)

    eng_label = er["label"]
    engagement_box.markdown(
        status_card("Engagement",
            f"{eng_label} · {er['source']}",
            alert   = eng_label in ["Low", "Very Low"],
            neutral = eng_label == "Detecting..."),
        unsafe_allow_html=True)

    hr_val = (f"YES — {hr['raise_type']} ({hr['hand_side']})"
              if hr["hand_raised"] else "NO")
    handrise_box.markdown(
        status_card("Hand Raised", hr_val, alert=hr["hand_raised"]),
        unsafe_allow_html=True)

    proxy_box.markdown(
        status_card("Proxy",
            "⚠ PROXY DETECTED" if pr["proxy_detected"] else "Normal",
            alert=pr["proxy_detected"]),
        unsafe_allow_html=True)

    gaze_lbl = (f"Looking Away ({pr['gaze_detail']})"
                if pr.get("gaze_detail") else "Looking Away")
    proxy_flags.markdown(
        proxy_flag_card(
            always_flags      = {"Multiple Faces": pr["multiple_faces"],
                                 "Static Face"   : pr["static_face"],
                                 gaze_lbl        : pr["looking_away"]},
            conditional_flags = {"Face Absent"   : pr["face_absent"]}
        ), unsafe_allow_html=True)

def render_waiting():
    for box in [stress_box, engagement_box, handrise_box, proxy_box]:
        box.markdown(status_card("—", "Waiting...", neutral=True),
                     unsafe_allow_html=True)
    proxy_flags.markdown(
        proxy_flag_card(always_flags={"Multiple Faces": False,
                                      "Static Face"   : False,
                                      "Looking Away"  : False}),
        unsafe_allow_html=True)

# =====================================================
#   ENGAGEMENT BACKGROUND THREAD
#   (same pattern as stress — non-blocking)
# =====================================================

global _engage_running
_engage_lock    = threading.Lock()
_engage_result  = {"label": "Detecting...", "confidence": 0, "source": "—", "mp_details": {}}
_engage_running = False

def _run_engagement(engage_det, frame, mesh_results):
    global _engage_result, _engage_running
    try:
        result = engage_det.update(frame, mesh_results)
        with _engage_lock:
            _engage_result = result
    finally:
        with _engage_lock:
            _engage_running = False

# =====================================================
#   MAIN LOOP
# =====================================================

if st.session_state.running:
    stress_det, engage_det, proxy_det, hand_det = load_detectors()
    pose, face_detection, face_mesh             = load_mediapipe()

    cap         = cv2.VideoCapture(0)
    frame_count = 0

    if not cap.isOpened():
        st.error("Cannot access webcam.")
        st.session_state.running = False
        st.stop()

    # Lower camera resolution for smoother streaming
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _engage_running = False

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

        # ---- MediaPipe every frame (fast, GPU-optimised) ----
        pose_results = pose.process(rgb)
        det_results  = face_detection.process(rgb)
        mesh_results = face_mesh.process(rgb)

        # ---- Hand Raise — every frame (instant) ----
        if handrise_enabled:
            st.session_state.handrise_result = hand_det.update(pose_results, h)

        # ---- Proxy — every frame (instant) ----
        if proxy_enabled:
            st.session_state.proxy_result = proxy_det.update(det_results, mesh_results)

        # ---- Stress — every frame, non-blocking (fires thread internally) ----
        if stress_enabled:
            st.session_state.stress_result = stress_det.update(frame, mesh_results)

        # ---- Engagement — background thread, fires every 10 frames ----
        if engagement_enabled and frame_count % 10 == 0:
            with _engage_lock:
                is_running = _engage_running
            if not is_running:
                with _engage_lock:
                    _engage_running = True
                t = threading.Thread(
                    target=_run_engagement,
                    args=(engage_det, frame.copy(), mesh_results),
                    daemon=True
                )
                t.start()
            with _engage_lock:
                st.session_state.engagement_result = _engage_result

        # ---- Draw overlays on frame (pure OpenCV, fast) ----
        display = draw_frame_overlays(
            frame.copy(),
            st.session_state.stress_result,
            st.session_state.engagement_result,
            st.session_state.handrise_result,
            st.session_state.proxy_result,
            mesh_results
        )

        # ---- Show frame — fixed width, no deprecated params ----
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        video_placeholder.image(display_rgb, channels="RGB", width=820)

        # ---- Update sidebar every N frames only ----
        if frame_count % SIDEBAR_UPDATE_EVERY == 0:
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