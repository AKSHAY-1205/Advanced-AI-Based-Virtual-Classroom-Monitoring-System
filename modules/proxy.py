import cv2
import numpy as np
import time
import mediapipe as mp

# ================= CONSTANTS =================

FACE_ABSENCE_LIMIT   = 5      # seconds with no FaceMesh landmarks → Face Absent
STATIC_FRAME_LIMIT   = 200    # frames frozen → Static Face
GAZE_FRAME_LIMIT     = 60     # frames looking away → Looking Away

# Face validity
EYE_WIDTH_RATIO_MIN  = 0.18
EYE_WIDTH_RATIO_MAX  = 0.58

# Head pose
YAW_THRESHOLD        = 0.08
PITCH_UP_THRESHOLD   = 0.10
PITCH_DOWN_THRESHOLD = 0.08

# Static detection
MOVEMENT_THRESHOLD   = 0.0015
EAR_BLINK_THRESHOLD  = 0.20
BLINK_RESET_FRAMES   = 4

# Proxy scoring
PROXY_SCORE_LIMIT    = 3

# Display
LABEL_X          = 20
LABEL_START_Y    = 40
LABEL_SPACING    = 38
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_FLAG  = 0.62
FONT_SCALE_PROXY = 0.95
THICKNESS_FLAG   = 2
THICKNESS_PROXY  = 3
COLOR_OK         = (30, 200, 30)
COLOR_WARN       = (0, 60, 255)
COLOR_PROXY_BG   = (0, 0, 180)
COLOR_WHITE      = (255, 255, 255)


# ================= HELPERS =================

def _ear(lm, p1, p2, p3, p4, p5, p6):
    v1 = np.linalg.norm([lm[p2].x - lm[p6].x, lm[p2].y - lm[p6].y])
    v2 = np.linalg.norm([lm[p3].x - lm[p5].x, lm[p3].y - lm[p5].y])
    h  = np.linalg.norm([lm[p1].x - lm[p4].x, lm[p1].y - lm[p4].y])
    return (v1 + v2) / (2.0 * h + 1e-6)


def _is_valid_face(mesh_landmark_obj):
    try:
        lm          = mesh_landmark_obj.landmark
        left_eye_x  = lm[33].x
        right_eye_x = lm[263].x
        eye_dist    = abs(right_eye_x - left_eye_x)
        all_x       = [l.x for l in lm]
        face_w      = max(all_x) - min(all_x)
        if face_w < 0.01:
            return False
        ratio = eye_dist / face_w
        return EYE_WIDTH_RATIO_MIN <= ratio <= EYE_WIDTH_RATIO_MAX
    except Exception:
        return True


# ================= PROXY DETECTOR =================

class ProxyDetector:
    """
    3 always-visible flags + 1 conditional flag:

    Always shown (green/red):
      1. Multiple Faces  — more than one valid face via FaceMesh
      2. Static Face     — face frozen (blink-aware)
      3. Looking Away    — head turned for N seconds

    Conditional (only shown when triggered):
      4. Face Absent     — shown ONLY when no face detected in frame
                           Hidden when face is present (even if still/static)

    Irregular Face: removed entirely.
    """

    def __init__(self):
        self.last_face_time   = time.time()
        self.prev_nose        = None
        self.static_frames    = 0
        self.blink_frames     = 0
        self.gaze_away_frames = 0
        self.gaze_detail      = ""

        self.multiple_faces   = False
        self.face_absent      = False
        self.static_face      = False
        self.looking_away     = False
        self.proxy_detected   = False
        self.proxy_score      = 0

    # --------------------------------------------------
    def update(self, detection_results, mesh_results):

        # ---- Validate faces via FaceMesh ----
        has_mesh = (
            mesh_results is not None and
            mesh_results.multi_face_landmarks is not None and
            len(mesh_results.multi_face_landmarks) > 0
        )

        valid_face_count = 0
        if has_mesh:
            for face_lm in mesh_results.multi_face_landmarks:
                if _is_valid_face(face_lm):
                    valid_face_count += 1

        real_face_present = valid_face_count > 0

        # ---- 1. Multiple Faces ----
        self.multiple_faces = valid_face_count > 1

        # ---- 2. Face Absent ----
        # Only True when FaceMesh finds zero valid faces for N seconds
        if real_face_present:
            self.last_face_time = time.time()
            self.face_absent    = False
        else:
            self.face_absent = (time.time() - self.last_face_time) > FACE_ABSENCE_LIMIT

        # ---- 3. Static Face + 4. Looking Away ----
        if real_face_present and has_mesh:
            lm = mesh_results.multi_face_landmarks[0].landmark

            # Blink check
            avg_ear     = (_ear(lm, 33, 160, 158, 133, 153, 144) +
                           _ear(lm, 362, 385, 387, 263, 373, 380)) / 2.0
            is_blinking = avg_ear < EAR_BLINK_THRESHOLD

            # Nose movement
            curr_nose = np.array([lm[1].x, lm[1].y])
            if self.prev_nose is not None:
                movement = np.linalg.norm(curr_nose - self.prev_nose)
                if is_blinking:
                    self.blink_frames += 1
                    if self.blink_frames >= BLINK_RESET_FRAMES:
                        self.static_frames = 0
                        self.blink_frames  = 0
                elif movement < MOVEMENT_THRESHOLD:
                    self.static_frames += 1
                    self.blink_frames   = 0
                else:
                    self.static_frames  = max(0, self.static_frames - 3)
                    self.blink_frames   = 0

            self.prev_nose   = curr_nose
            self.static_face = self.static_frames > STATIC_FRAME_LIMIT

            # Head pose
            yaw_diff   = abs(
                abs(lm[1].x - lm[234].x) - abs(lm[1].x - lm[454].x)
            )
            face_h     = abs(lm[10].y - lm[152].y)
            pitch_norm = (
                (lm[1].y - (lm[10].y + lm[152].y) / 2.0) / face_h
                if face_h > 0 else 0
            )

            turned_side  = yaw_diff > YAW_THRESHOLD
            looking_up   = pitch_norm < -PITCH_UP_THRESHOLD
            looking_down = pitch_norm > PITCH_DOWN_THRESHOLD
            is_away      = turned_side or looking_up or looking_down

            if is_away:
                self.gaze_away_frames += 1
                self.gaze_detail = (
                    "Side" if turned_side else
                    "Up"   if looking_up  else
                    "Down"
                )
            else:
                self.gaze_away_frames = max(0, self.gaze_away_frames - 3)
                if self.gaze_away_frames == 0:
                    self.gaze_detail = ""

            self.looking_away = self.gaze_away_frames > GAZE_FRAME_LIMIT

        else:
            self.prev_nose        = None
            self.static_frames    = 0
            self.blink_frames     = 0
            self.static_face      = False
            self.looking_away     = False
            self.gaze_detail      = ""
            self.gaze_away_frames = 0

        # ---- Proxy Score ----
        self.proxy_score = 0
        if self.multiple_faces: self.proxy_score += 3
        if self.face_absent:    self.proxy_score += 3
        if self.static_face:    self.proxy_score += 3
        if self.looking_away:   self.proxy_score += 2

        self.proxy_detected = self.proxy_score >= PROXY_SCORE_LIMIT

        return self._build_result()

    # --------------------------------------------------
    def _build_result(self):
        return {
            "proxy_detected": self.proxy_detected,
            "proxy_score"   : self.proxy_score,
            "multiple_faces": self.multiple_faces,
            "face_absent"   : self.face_absent,
            "static_face"   : self.static_face,
            "looking_away"  : self.looking_away,
            "gaze_detail"   : self.gaze_detail
        }

    # --------------------------------------------------
    def draw(self, frame, result):
        """
        Drawing rules:
          - Multiple Faces : always shown (green/red)
          - Static Face    : always shown (green/red)
          - Looking Away   : always shown (green/red)
          - Face Absent    : ONLY drawn when triggered (not shown when OK)
        """
        gaze_label = "Looking Away"
        if result.get("gaze_detail"):
            gaze_label = f"Looking Away ({result['gaze_detail']})"

        # Always-on flags
        always_flags = [
            ("Multiple Faces", result["multiple_faces"]),
            ("Static Face",    result["static_face"]),
            (gaze_label,       result["looking_away"]),
        ]

        for i, (label, triggered) in enumerate(always_flags):
            y      = LABEL_START_Y + i * LABEL_SPACING
            color  = COLOR_WARN if triggered else COLOR_OK
            status = "DETECTED" if triggered else "OK"
            text   = f"{label}: {status}"

            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE_FLAG, THICKNESS_FLAG)
            cv2.rectangle(frame,
                (LABEL_X - 6, y - th - 4),
                (LABEL_X + tw + 6, y + 4),
                (20, 20, 20), -1)
            cv2.putText(frame, text, (LABEL_X, y),
                        FONT, FONT_SCALE_FLAG, COLOR_WARN if triggered else COLOR_OK,
                        THICKNESS_FLAG)

        # Face Absent — only drawn when triggered
        next_y = LABEL_START_Y + len(always_flags) * LABEL_SPACING

        if result["face_absent"]:
            text = "Face Absent: DETECTED"
            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE_FLAG, THICKNESS_FLAG)
            cv2.rectangle(frame,
                (LABEL_X - 6, next_y - th - 4),
                (LABEL_X + tw + 6, next_y + 4),
                (20, 20, 20), -1)
            cv2.putText(frame, text, (LABEL_X, next_y),
                        FONT, FONT_SCALE_FLAG, COLOR_WARN, THICKNESS_FLAG)
            next_y += LABEL_SPACING

        # Proxy banner
        if result["proxy_detected"]:
            banner_y    = next_y + 10
            banner_text = "!! PROXY DETECTED !!"
            (bw, bh), _ = cv2.getTextSize(
                banner_text, FONT, FONT_SCALE_PROXY, THICKNESS_PROXY)
            cv2.rectangle(frame,
                (LABEL_X - 8, banner_y - bh - 6),
                (LABEL_X + bw + 8, banner_y + 6),
                COLOR_PROXY_BG, -1)
            cv2.putText(frame, banner_text, (LABEL_X, banner_y),
                        FONT, FONT_SCALE_PROXY, COLOR_WHITE, THICKNESS_PROXY)

        return frame


# ================= STANDALONE TEST =================

if __name__ == "__main__":
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh      = mp.solutions.face_mesh

    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6)

    detector = ProxyDetector()
    cap      = cv2.VideoCapture(0)

    print("Proxy Detection — Press Q to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame        = cv2.flip(frame, 1)
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det_results  = face_detection.process(rgb)
        mesh_results = face_mesh.process(rgb)
        result       = detector.update(det_results, mesh_results)
        frame        = detector.draw(frame, result)
        cv2.imshow("Proxy Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()