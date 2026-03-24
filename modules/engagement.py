import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ================= CONSTANTS =================

IMG_SIZE         = 160
HIGH_CONF        = 0.70
MID_CONF         = 0.50

YAW_THRESHOLD    = 0.07
PITCH_THRESHOLD  = 0.06
EAR_THRESHOLD    = 0.22

CLASS_LABELS = {0: "Very Low", 1: "Low", 2: "High", 3: "Very High"}
ENGAGEMENT_SCORES = {"Very Low": 0, "Low": 1, "High": 2, "Very High": 3}
SCORE_TO_LABEL    = {0: "Very Low", 1: "Low", 2: "High", 3: "Very High"}

FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.75
THICKNESS        = 2
LABEL_X          = 20
LABEL_Y          = 330

COLOR_VERY_HIGH  = (0, 220, 0)
COLOR_HIGH       = (30, 180, 30)
COLOR_LOW        = (0, 140, 255)
COLOR_VERY_LOW   = (0, 60, 255)
COLOR_DETECTING  = (200, 200, 200)


# ================= COMPATIBILITY LOADER =================

def _build_engagement_architecture():
    """Rebuild EfficientNetB0 engagement model architecture."""
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base   = EfficientNetB0(weights=None, include_top=False, input_shape=(160, 160, 3))
    x      = base.output
    x      = GlobalAveragePooling2D()(x)
    x      = Dense(256, activation='relu')(x)
    x      = Dropout(0.4)(x)
    output = Dense(4, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)


def load_model_compatible(model_path, architecture_fn):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("  Model loaded normally.")
        return model
    except Exception as e:
        if "batch_shape" in str(e) or "Unrecognized keyword arguments" in str(e):
            print("  Version mismatch detected — loading weights only...")
            model = architecture_fn()
            model.load_weights(model_path, by_name=False, skip_mismatch=False)
            print("  Weights loaded into rebuilt architecture.")
            return model
        else:
            raise e


# ================= ENGAGEMENT DETECTOR =================

class EngagementDetector:
    """
    Hybrid engagement detector:
      - EfficientNetB0 model for learned prediction
      - MediaPipe FaceMesh for head pose + EAR heuristics
    Fuses both based on model confidence level.
    """

    def __init__(self, model_path: str):
        print(f"Loading engagement model from {model_path} ...")
        self.model      = load_model_compatible(model_path, _build_engagement_architecture)
        self.label      = "Detecting..."
        self.confidence = 0.0
        self.source     = "—"
        print("Engagement model ready.")

    def _preprocess(self, frame):
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    def _mediapipe_score(self, mesh_results):
        if not mesh_results or not mesh_results.multi_face_landmarks:
            return 0, {"gaze": "no face", "ear": 0.0}

        lm = mesh_results.multi_face_landmarks[0].landmark

        # --- Head Pose ---
        nose        = lm[1]
        left_cheek  = lm[234]
        right_cheek = lm[454]
        forehead    = lm[10]
        chin        = lm[152]

        yaw_diff   = abs(abs(nose.x - left_cheek.x) - abs(nose.x - right_cheek.x))
        face_h     = abs(forehead.y - chin.y)
        pitch_diff = abs(nose.y - (forehead.y + chin.y) / 2) / face_h if face_h > 0 else 0

        on_screen  = (yaw_diff < YAW_THRESHOLD) and (pitch_diff < PITCH_THRESHOLD)

        # --- Eye Aspect Ratio ---
        def ear(p1, p2, p3, p4, p5, p6):
            v1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
            v2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
            h  = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])
            return (v1 + v2) / (2.0 * h + 1e-6)

        left_ear  = ear(lm[33],  lm[160], lm[158], lm[133], lm[153], lm[144])
        right_ear = ear(lm[362], lm[385], lm[387], lm[263], lm[373], lm[380])
        avg_ear   = (left_ear + right_ear) / 2.0
        eyes_open = avg_ear > EAR_THRESHOLD

        if on_screen and eyes_open:
            score = 3
        elif on_screen or eyes_open:
            score = 2
        elif not on_screen and eyes_open:
            score = 1
        else:
            score = 0

        return score, {"gaze": "on screen" if on_screen else "away",
                       "ear": round(float(avg_ear), 3), "eyes_open": eyes_open}

    def update(self, frame, mesh_results):
        inp         = self._preprocess(frame)
        probs       = self.model.predict(inp, verbose=0)[0]
        model_class = int(np.argmax(probs))
        model_conf  = float(probs[model_class])
        model_score = ENGAGEMENT_SCORES[CLASS_LABELS[model_class]]

        mp_score, mp_details = self._mediapipe_score(mesh_results)

        if model_conf >= HIGH_CONF:
            final_score = model_score
            source      = f"Model ({int(model_conf*100)}%)"
        elif model_conf >= MID_CONF:
            final_score = round((model_score + mp_score) / 2.0)
            source      = f"Blended ({int(model_conf*100)}%)"
        else:
            final_score = round(mp_score * 0.7 + model_score * 0.3)
            source      = f"({int(model_conf*100)}%)"

        final_score     = max(0, min(3, final_score))
        self.label      = SCORE_TO_LABEL[final_score]
        self.confidence = model_conf
        self.source     = source

        return self._build_result(mp_details)

    def _build_result(self, mp_details=None):
        return {
            "label"     : self.label,
            "confidence": round(self.confidence * 100, 1),
            "source"    : self.source,
            "mp_details": mp_details or {}
        }

    def draw(self, frame, result):
        label  = result["label"]
        source = result["source"]
        color  = {
            "Very High"   : COLOR_VERY_HIGH,
            "High"        : COLOR_HIGH,
            "Low"         : COLOR_LOW,
            "Very Low"    : COLOR_VERY_LOW,
            "Detecting...": COLOR_DETECTING
        }.get(label, COLOR_DETECTING)

        text = f"Engagement: {label}"
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame,
            (LABEL_X - 6, LABEL_Y - th - 4),
            (LABEL_X + tw + 6, LABEL_Y + 4),
            (20, 20, 20), -1)
        cv2.putText(frame, text, (LABEL_X, LABEL_Y),
                    FONT, FONT_SCALE, color, THICKNESS)
        cv2.putText(frame, f"  via {source}", (LABEL_X, LABEL_Y + 22),
                    FONT, 0.45, (180, 180, 180), 1)
        return frame


# ================= STANDALONE TEST =================

if __name__ == "__main__":
    import os
    import mediapipe as mp

    MODEL_PATH   = "models\\engagement_model_best.h5"
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6)

    detector    = EngagementDetector(MODEL_PATH)
    cap         = cv2.VideoCapture(0)
    frame_count = 0
    last_result = detector._build_result()

    print("Engagement Detection — Press Q to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame       = cv2.flip(frame, 1)
        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = face_mesh.process(rgb)
        frame_count += 1
        if frame_count % 5 == 0:
            last_result = detector.update(frame, mesh_result)
        frame = detector.draw(frame, last_result)
        cv2.imshow("Engagement Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()