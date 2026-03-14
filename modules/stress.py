import cv2
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= CONSTANTS =================

IMG_SIZE         = 224
MODEL_THRESHOLD  = 0.65
SMOOTHING_FRAMES = 8

# Stress requires BOTH:
#   a) model prediction > MODEL_THRESHOLD
#   b) at least MIN_CUES MediaPipe stress cues firing simultaneously
# This prevents false positives from either signal alone
MIN_CUES_FOR_STRESS = 2      # minimum cues needed alongside model signal
MP_ONLY_THRESHOLD   = 4      # if 4+ cues fire, stress even without model

# ---- MediaPipe thresholds — conservative to avoid false triggers ----

# Eye squint: eyes genuinely narrowed (not just natural eye shape)
# Normal relaxed EAR ~ 0.28–0.35. Squinting < 0.18 is clearly visible
EYE_SQUINT_THRESHOLD   = 0.18   # deliberately low — only catch real squinting

# Brow furrow: inner brows genuinely close
# Normal brow distance ~ 0.10–0.15 of face width
BROW_FURROW_THRESHOLD  = 0.075  # only trigger on clear furrowing

# Mouth open (tension/distress): clearly open mouth
MOUTH_OPEN_THRESHOLD   = 0.10   # 10% of face height — clearly open

# Lip compression: very tight pressed lips
LIP_THIN_THRESHOLD     = 0.020  # only very compressed lips

# Nose flare: clearly wider than normal
NOSE_FLARE_THRESHOLD   = 0.22

FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.75
THICKNESS        = 2
COLOR_STRESS     = (0, 60, 255)
COLOR_NO_STRESS  = (30, 200, 30)
COLOR_DETECTING  = (200, 200, 200)
LABEL_X          = 20
LABEL_Y          = 285


# ================= COMPATIBILITY LOADER =================

def _build_stress_architecture():
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    base   = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x      = base.output
    x      = GlobalAveragePooling2D()(x)
    x      = Dense(128, activation='relu')(x)
    x      = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=output)


def load_model_compatible(model_path, architecture_fn):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("  Model loaded normally.")
        return model
    except Exception as e:
        if "batch_shape" in str(e) or "Unrecognized keyword arguments" in str(e):
            print("  Version mismatch — loading weights only...")
            model = architecture_fn()
            model.load_weights(model_path, by_name=False, skip_mismatch=False)
            print("  Weights loaded into rebuilt architecture.")
            return model
        else:
            raise e


# ================= EXPRESSION ANALYSER =================

class _ExpressionAnalyser:
    """
    Counts how many stress cues are simultaneously active.
    Returns (cue_count: int, cues: list).
    Uses conservative thresholds — only fires on clearly visible expressions.
    """

    def _ear(self, lm, p1, p2, p3, p4, p5, p6):
        v1 = np.linalg.norm([lm[p2].x - lm[p6].x, lm[p2].y - lm[p6].y])
        v2 = np.linalg.norm([lm[p3].x - lm[p5].x, lm[p3].y - lm[p5].y])
        h  = np.linalg.norm([lm[p1].x - lm[p4].x, lm[p1].y - lm[p4].y])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def analyse(self, mesh_results):
        """
        Returns (cue_count: int, cues_fired: list)
        0 cues = calm expression. 2+ cues = stress expression visible.
        """
        if not mesh_results or not mesh_results.multi_face_landmarks:
            return 0, []

        lm     = mesh_results.multi_face_landmarks[0].landmark
        cues   = []

        face_h = abs(lm[10].y - lm[152].y)
        face_w = abs(lm[454].x - lm[234].x)
        if face_h < 0.01 or face_w < 0.01:
            return 0, []

        # 1. Eye squint — both eyes clearly narrowed
        left_ear  = self._ear(lm, 33,  160, 158, 133, 153, 144)
        right_ear = self._ear(lm, 362, 385, 387, 263, 373, 380)
        avg_ear   = (left_ear + right_ear) / 2.0
        if avg_ear < EYE_SQUINT_THRESHOLD:
            cues.append("eye_squint")

        # 2. Brow furrow — clearly pulled together
        brow_dist = abs(lm[285].x - lm[55].x) / face_w
        if brow_dist < BROW_FURROW_THRESHOLD:
            cues.append("brow_furrow")

        # 3. Mouth open (distress/tension)
        mouth_open = abs(lm[17].y - lm[0].y) / face_h
        if mouth_open > MOUTH_OPEN_THRESHOLD:
            cues.append("mouth_open")

        # 4. Lip compression
        lip_gap = abs(lm[14].y - lm[13].y) / face_h
        if lip_gap < LIP_THIN_THRESHOLD:
            cues.append("lip_compress")

        # 5. Nose flare
        alar_w = abs(lm[358].x - lm[129].x) / face_w
        if alar_w > NOSE_FLARE_THRESHOLD:
            cues.append("nose_flare")

        return len(cues), cues


# ================= STRESS DETECTOR =================

class StressDetector:
    """
    Threaded stress detector — model inference runs in background thread
    so it NEVER blocks the video loop.

    Decision logic (prevents false positives):
      STRESS if:
        (model_pred > 0.65  AND  cue_count >= 2)   ← both agree
        OR
        (cue_count >= 4)                             ← overwhelming expression signal

      NO STRESS otherwise — single cue alone never triggers stress

    Model preprocessing exactly matches working notebook:
      full frame, float32, preprocess_input, no BGR→RGB conversion.
    """

    def __init__(self, model_path: str):
        print(f"Loading stress model from {model_path} ...")
        self.model    = load_model_compatible(model_path, _build_stress_architecture)
        self.analyser = _ExpressionAnalyser()

        # Smoothing counters
        self.stress_counter    = 0
        self.no_stress_counter = 0
        self.label             = "Detecting..."
        self.confidence        = 0.0

        # Threading — model runs in background
        self._lock          = threading.Lock()
        self._model_pred    = 0.0      # latest model prediction
        self._inferring     = False    # is inference currently running?
        self._pending_frame = None     # next frame to infer

        print("Stress model ready.")

    # --------------------------------------------------
    def _preprocess(self, frame):
        """Exactly matches working notebook preprocess_frame()."""
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    # --------------------------------------------------
    def _run_inference(self, frame):
        """Runs in background thread. Updates _model_pred when done."""
        try:
            inp  = self._preprocess(frame)
            pred = float(self.model.predict(inp, verbose=0)[0][0])
            with self._lock:
                self._model_pred = pred
        finally:
            with self._lock:
                self._inferring = False

    # --------------------------------------------------
    def update(self, frame, mesh_results=None):
        """
        Call every frame — never blocks.
        Fires background inference thread when previous one finishes.
        Uses last known model prediction between inference calls.

        Args:
            frame        : raw BGR full frame
            mesh_results : FaceMesh results from app.py

        Returns:
            dict with label and confidence
        """

        # ---- Fire background inference if idle ----
        with self._lock:
            if not self._inferring:
                self._inferring     = True
                frame_copy          = frame.copy()
                t = threading.Thread(
                    target=self._run_inference,
                    args=(frame_copy,),
                    daemon=True
                )
                t.start()

        # ---- Read last known model prediction (non-blocking) ----
        with self._lock:
            model_pred = self._model_pred

        # ---- MediaPipe expression cues ----
        cue_count, _ = self.analyser.analyse(mesh_results)

        # ---- Decision logic ----
        # Both model AND expressions must agree (prevents false positives)
        model_says_stress = model_pred > MODEL_THRESHOLD
        strong_expression = cue_count >= MIN_CUES_FOR_STRESS
        overwhelming_cues = cue_count >= MP_ONLY_THRESHOLD

        is_stress = (model_says_stress and strong_expression) or overwhelming_cues

        # ---- Smoothing ----
        if is_stress:
            self.stress_counter    += 1
            self.no_stress_counter  = 0
        else:
            self.no_stress_counter += 1
            self.stress_counter     = 0

        if self.stress_counter >= SMOOTHING_FRAMES:
            self.label      = "STRESS"
            self.confidence = model_pred
        elif self.no_stress_counter >= SMOOTHING_FRAMES:
            self.label      = "NO STRESS"
            self.confidence = 1.0 - model_pred

        return self._build_result()

    # --------------------------------------------------
    def _build_result(self):
        return {
            "label"     : self.label,
            "confidence": round(self.confidence * 100, 1)
        }

    # --------------------------------------------------
    def draw(self, frame, result):
        """Used in standalone test only."""
        label = result["label"]
        conf  = result["confidence"]
        color = (COLOR_STRESS    if label == "STRESS"    else
                 COLOR_NO_STRESS if label == "NO STRESS" else COLOR_DETECTING)
        text = f"Stress: {label} ({conf}%)"
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame,
            (LABEL_X-6, LABEL_Y-th-4), (LABEL_X+tw+6, LABEL_Y+4), (20,20,20), -1)
        cv2.putText(frame, text, (LABEL_X, LABEL_Y), FONT, FONT_SCALE, color, THICKNESS)
        return frame


# ================= STANDALONE TEST =================

if __name__ == "__main__":
    import os
    import mediapipe as mp

    MODEL_PATH   = os.path.join("..", "models", "stress_model_da.h5")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6)

    detector    = StressDetector(MODEL_PATH)
    cap         = cv2.VideoCapture(0)
    last_result = detector._build_result()

    print("Stress Detection — Press Q to quit")
    print("Triggers on: eye squint + brow furrow, or 4+ cues simultaneously")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame       = cv2.flip(frame, 1)
        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = face_mesh.process(rgb)
        last_result = detector.update(frame, mesh_result)
        frame       = detector.draw(frame, last_result)
        cv2.imshow("Stress Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()