import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= CONSTANTS =================
# These match EXACTLY the working real-time notebook

IMG_SIZE         = 224
THRESHOLD        = 0.65      # same as working notebook
SMOOTHING_FRAMES = 10        # same as working notebook

# MediaPipe secondary signal weight
# Model is primary — MediaPipe only nudges when model is borderline
# If model pred is 0.55–0.75 (borderline), MediaPipe can shift decision
BORDERLINE_LOW   = 0.55
BORDERLINE_HIGH  = 0.75
MP_NUDGE         = 0.08      # how much MediaPipe shifts score when borderline

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


# ================= MEDIAPIPE STRESS CUES (silent) =================

class _FacialStressAnalyser:
    """
    Analyses FaceMesh landmarks for stress-related facial expressions.
    Used as a SECONDARY signal only — nudges borderline model predictions.
    NEVER shown on screen.

    Stress cues checked:
      1. Lip compression   — tight lips (reduced inner lip gap)
      2. Brow furrow       — inner brows pulled together
      3. Jaw clench        — minimal mouth opening
      4. Mouth corner drop — corners pulled down
      5. Nose flare        — alar width increase
    """

    # Landmark-based thresholds
    LIP_COMPRESS_THRESHOLD = 0.025
    BROW_FURROW_THRESHOLD  = 0.085
    JAW_CLENCH_THRESHOLD   = 0.018
    MOUTH_CORNER_DROP      = 0.005
    NOSE_FLARE_THRESHOLD   = 0.20

    def analyse(self, mesh_results):
        """
        Returns stress_score in [0.0, 1.0].
        0.0 = no stress cues, 1.0 = all stress cues firing.
        Returns 0.5 (neutral) if no face found — won't influence model.
        """
        if not mesh_results or not mesh_results.multi_face_landmarks:
            return 0.5   # neutral — no influence on model

        lm     = mesh_results.multi_face_landmarks[0].landmark
        score  = 0.0
        cues   = 0

        # Reference measurements
        face_h = abs(lm[10].y - lm[152].y)
        face_w = abs(lm[454].x - lm[234].x)

        if face_h < 0.01 or face_w < 0.01:
            return 0.5

        # 1. Lip compression
        lip_gap = abs(lm[14].y - lm[13].y) / face_h
        if lip_gap < self.LIP_COMPRESS_THRESHOLD:
            score += 0.25
            cues  += 1

        # 2. Brow furrow
        brow_dist = abs(lm[285].x - lm[55].x) / face_w
        if brow_dist < self.BROW_FURROW_THRESHOLD:
            score += 0.25
            cues  += 1

        # 3. Jaw clench
        mouth_open = abs(lm[17].y - lm[0].y) / face_h
        if mouth_open < self.JAW_CLENCH_THRESHOLD:
            score += 0.20
            cues  += 1

        # 4. Mouth corner drop
        mouth_mid_y = (lm[0].y + lm[17].y) / 2.0
        if (lm[61].y - mouth_mid_y > self.MOUTH_CORNER_DROP or
                lm[291].y - mouth_mid_y > self.MOUTH_CORNER_DROP):
            score += 0.15
            cues  += 1

        # 5. Nose flare
        alar_w = abs(lm[358].x - lm[129].x) / face_w
        if alar_w > self.NOSE_FLARE_THRESHOLD:
            score += 0.15
            cues  += 1

        return min(1.0, score)


# ================= STRESS DETECTOR =================

class StressDetector:
    """
    Stress detection matching the working real-time notebook exactly:
      - Full frame passed to model (NO face crop — model trained on full frames)
      - preprocess_input scaling (matches notebook inference)
      - Threshold 0.65, smoothing 10 frames

    MediaPipe facial cues act as a SECONDARY nudge only:
      - Only applied when model prediction is borderline (0.55–0.75)
      - If model is clearly STRESS (>0.75) or clearly NO STRESS (<0.55),
        MediaPipe has zero influence
      - Nothing about MediaPipe shown on screen
    """

    def __init__(self, model_path: str):
        print(f"Loading stress model from {model_path} ...")
        self.model             = load_model_compatible(model_path, _build_stress_architecture)
        self.mp_analyser       = _FacialStressAnalyser()
        self.stress_counter    = 0
        self.no_stress_counter = 0
        self.label             = "Detecting..."
        self.confidence        = 0.0
        print("Stress model ready.")

    # --------------------------------------------------
    def _preprocess(self, frame):
        """
        Exactly matches the working notebook's preprocess_frame():
          - resize to 224x224
          - float32 cast
          - preprocess_input (scales to [-1, 1])
          - expand dims
        NO color conversion — notebook passes BGR frame directly.
        """
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    # --------------------------------------------------
    def update(self, frame, mesh_results=None):
        """
        Run stress inference on full frame.
        Call every N frames (recommended: every 5).

        Args:
            frame        : raw BGR frame (full frame, not cropped)
            mesh_results : FaceMesh results — used silently for borderline nudge

        Returns:
            dict with label and confidence
        """

        # ---- Primary: model on full frame ----
        inp        = self._preprocess(frame)
        prediction = float(self.model.predict(inp, verbose=0)[0][0])

        # ---- Secondary: MediaPipe nudge (borderline only) ----
        # Only adjusts prediction slightly when model is unsure
        if BORDERLINE_LOW <= prediction <= BORDERLINE_HIGH:
            mp_score = self.mp_analyser.analyse(mesh_results)
            # mp_score > 0.5 means stress cues present → nudge up
            # mp_score < 0.5 means calm cues → nudge down
            nudge      = (mp_score - 0.5) * MP_NUDGE
            prediction = float(np.clip(prediction + nudge, 0.0, 1.0))

        # ---- Smoothing (same as notebook) ----
        if prediction > THRESHOLD:
            self.stress_counter    += 1
            self.no_stress_counter  = 0
        else:
            self.no_stress_counter += 1
            self.stress_counter     = 0

        if self.stress_counter >= SMOOTHING_FRAMES:
            self.label      = "STRESS"
            self.confidence = prediction
        elif self.no_stress_counter >= SMOOTHING_FRAMES:
            self.label      = "NO STRESS"
            self.confidence = 1.0 - prediction

        return self._build_result()

    # --------------------------------------------------
    def _build_result(self):
        return {
            "label"     : self.label,
            "confidence": round(self.confidence * 100, 1)
        }

    # --------------------------------------------------
    def draw(self, frame, result):
        """
        Draw stress label on frame.
        Only shows STRESS / NO STRESS — no MediaPipe details.
        """
        label = result["label"]
        conf  = result["confidence"]

        color = (COLOR_STRESS    if label == "STRESS"     else
                 COLOR_NO_STRESS if label == "NO STRESS"  else
                 COLOR_DETECTING)

        text = f"Stress: {label} ({conf}%)"
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame,
            (LABEL_X - 6, LABEL_Y - th - 4),
            (LABEL_X + tw + 6, LABEL_Y + 4),
            (20, 20, 20), -1)
        cv2.putText(frame, text, (LABEL_X, LABEL_Y),
                    FONT, FONT_SCALE, color, THICKNESS)
        return frame


# ================= STANDALONE TEST =================

if __name__ == "__main__":
    import os
    import mediapipe as mp

    MODEL_PATH   = "models\\stress_model_da.h5"
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    )

    detector    = StressDetector(MODEL_PATH)
    cap         = cv2.VideoCapture(0)
    frame_count = 0
    last_result = detector._build_result()

    print("Stress Detection — Press Q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame        = cv2.flip(frame, 1)
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result  = face_mesh.process(rgb)
        frame_count += 1

        if frame_count % 5 == 0:
            last_result = detector.update(frame, mesh_result)

        frame = detector.draw(frame, last_result)
        cv2.imshow("Stress Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()