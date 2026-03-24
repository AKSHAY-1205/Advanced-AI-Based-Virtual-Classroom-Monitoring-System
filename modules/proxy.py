import cv2
import numpy as np
import time
import mediapipe as mp


FACE_ABSENCE_LIMIT   = 5
STATIC_FRAME_LIMIT   = 200
GAZE_FRAME_LIMIT     = 60

YAW_THRESHOLD        = 0.08
PITCH_UP_THRESHOLD   = 0.10
PITCH_DOWN_THRESHOLD = 0.08

MOVEMENT_THRESHOLD   = 0.0015
EAR_BLINK_THRESHOLD  = 0.20
BLINK_RESET_FRAMES   = 4

PROXY_SCORE_LIMIT    = 3

LABEL_X          = 20
LABEL_START_Y    = 40
LABEL_SPACING    = 34
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.60
FONT_SCALE_PROXY = 0.90
THICKNESS        = 2
COLOR_OK         = (30, 200, 30)
COLOR_WARN       = (0, 60, 255)
COLOR_PROXY_BG   = (0, 0, 180)
COLOR_WHITE      = (255, 255, 255)



def _ear(lm, p1, p2, p3, p4, p5, p6):
    v1 = np.linalg.norm([lm[p2].x - lm[p6].x, lm[p2].y - lm[p6].y])
    v2 = np.linalg.norm([lm[p3].x - lm[p5].x, lm[p3].y - lm[p5].y])
    h  = np.linalg.norm([lm[p1].x - lm[p4].x, lm[p1].y - lm[p4].y])
    return (v1 + v2) / (2.0 * h + 1e-6)


class ProxyDetector:
    """
    3 always-visible flags + 1 conditional:

    Always shown (green OK / red DETECTED):
      1. Multiple Faces  — FaceMesh finds >1 face
      2. Static Face     — face landmarks frozen for N frames (blink-aware)
      3. Looking Away    — head turned side/up/down for N seconds

    Conditional (only shown when triggered):
      4. Face Absent     — FaceMesh returns ZERO landmarks for N seconds
                           This is the ONLY condition. No face = no landmarks.
                           If face is present in any form, this stays False.

    Face validity check REMOVED — it was causing false "face absent" triggers
    on real faces. Now we trust FaceMesh directly: if it returns landmarks,
    a face is present. FaceMesh itself is robust enough to reject non-faces.
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

    def update(self, detection_results, mesh_results):

        face_count = 0
        if (mesh_results is not None and
                mesh_results.multi_face_landmarks is not None):
            face_count = len(mesh_results.multi_face_landmarks)

        face_present = face_count > 0

        # ---- 1. Multiple Faces ----
        self.multiple_faces = face_count > 1

        # ---- 2. Face Absent ----
        # ONLY when FaceMesh returns zero landmarks for N seconds.
        # If your face is in frame — even turned, even still — FaceMesh
        # will return landmarks and this stays False.
        if face_present:
            self.last_face_time = time.time()
            self.face_absent    = False
        else:
            self.face_absent = (time.time() - self.last_face_time) > FACE_ABSENCE_LIMIT

        # ---- 3. Static Face + 4. Looking Away ----
        if face_present:
            lm = mesh_results.multi_face_landmarks[0].landmark

            # Blink check — alive indicator
            avg_ear     = (_ear(lm, 33,  160, 158, 133, 153, 144) +
                           _ear(lm, 362, 385, 387, 263, 373, 380)) / 2.0
            is_blinking = avg_ear < EAR_BLINK_THRESHOLD

            # Nose movement — static detection
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
            yaw_diff   = abs(abs(lm[1].x - lm[234].x) - abs(lm[1].x - lm[454].x))
            face_h     = abs(lm[10].y - lm[152].y)
            pitch_norm = ((lm[1].y - (lm[10].y + lm[152].y) / 2.0) / face_h
                          if face_h > 0 else 0)

            turned_side  = yaw_diff > YAW_THRESHOLD
            looking_up   = pitch_norm < -PITCH_UP_THRESHOLD
            looking_down = pitch_norm > PITCH_DOWN_THRESHOLD
            is_away      = turned_side or looking_up or looking_down

            if is_away:
                self.gaze_away_frames += 1
                self.gaze_detail = ("Side" if turned_side else
                                    "Up"   if looking_up  else "Down")
            else:
                self.gaze_away_frames = max(0, self.gaze_away_frames - 3)
                if self.gaze_away_frames == 0:
                    self.gaze_detail = ""

            self.looking_away = self.gaze_away_frames > GAZE_FRAME_LIMIT

        else:
            # Reset all tracking when no face
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
        """Used in standalone test. app.py uses PIL-based draw_frame_labels instead."""
        gaze_label = (f"Looking Away ({result['gaze_detail']})"
                      if result.get("gaze_detail") else "Looking Away")

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
            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
            cv2.rectangle(frame, (LABEL_X-6, y-th-4), (LABEL_X+tw+6, y+4), (20,20,20), -1)
            cv2.putText(frame, text, (LABEL_X, y), FONT, FONT_SCALE, color, THICKNESS)

        next_y = LABEL_START_Y + len(always_flags) * LABEL_SPACING

        if result["face_absent"]:
            text = "Face Absent: DETECTED"
            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
            cv2.rectangle(frame, (LABEL_X-6, next_y-th-4), (LABEL_X+tw+6, next_y+4), (20,20,20), -1)
            cv2.putText(frame, text, (LABEL_X, next_y), FONT, FONT_SCALE, COLOR_WARN, THICKNESS)
            next_y += LABEL_SPACING

        if result["proxy_detected"]:
            banner_y    = next_y + 8
            banner_text = "!! PROXY DETECTED !!"
            (bw, bh), _ = cv2.getTextSize(banner_text, FONT, FONT_SCALE_PROXY, 3)
            cv2.rectangle(frame, (LABEL_X-8, banner_y-bh-6), (LABEL_X+bw+8, banner_y+6),
                          COLOR_PROXY_BG, -1)
            cv2.putText(frame, banner_text, (LABEL_X, banner_y),
                        FONT, FONT_SCALE_PROXY, COLOR_WHITE, 3)
        return frame


# ================= STANDALONE TEST =================

if __name__ == "__main__":
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh      = mp.solutions.face_mesh
    face_detection    = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6)
    face_mesh         = mp_face_mesh.FaceMesh(
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