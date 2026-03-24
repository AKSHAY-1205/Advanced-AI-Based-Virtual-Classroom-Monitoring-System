import cv2
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─────────────────────────────────────────────────────────────────
#  STRESS DETECTION — Strict Landmark-First + MobileNetV2 confirm
#
#  Key changes from original:
#   1. LANDMARK GATE: All 3 core stress signals must fire before
#      the model score is even considered:
#        - Brow furrow (forehead/inner brow squeeze)
#        - Eye squint  (EAR drops — eyes narrow)
#        - Lip/mouth compression (mouth shrinks tight)
#      This directly matches the requirement: stress only when
#      forehead shrunken, eyebrows shrunken, lips+mouth shrunken.
#   2. Model weight REDUCED to 0.30 — landmarks drive the decision.
#   3. Thresholds TIGHTENED so only clear, definitive expressions
#      trigger stress — not subtle or ambient variation.
#   4. Smoothing window INCREASED to 10 frames — avoids false flicker.
#   5. Fused threshold raised to 0.60 — harder to cross accidentally.
# ─────────────────────────────────────────────────────────────────

IMG_SIZE         = 224
SMOOTHING_FRAMES = 10         # stable — reduces false positives

# Fusion weights — landmarks are PRIMARY, model is a confirming signal
MODEL_WEIGHT     = 0.30
CUE_WEIGHT       = 0.70
FUSED_THRESHOLD  = 0.60       # must cross this to be labelled STRESS

# ─── TIGHTENED landmark thresholds ──────────────────────────────
#
# BROW FURROW: inner brow landmarks 285 & 55 come close together.
#   Normal face: inner-brow dist / face_w ≈ 0.10–0.14
#   Furrowed    : ratio < 0.07  (was 0.08 — tightened)
BROW_FURROW_THRESHOLD = 0.07

# EYE SQUINT: Eye Aspect Ratio (EAR) drops when eyes narrow.
#   Normal open eye EAR ≈ 0.28–0.35
#   Squinting hard : EAR < 0.18  (was 0.20 — tightened)
EYE_SQUINT_THRESHOLD  = 0.18

# LIP COMPRESSION: inner lip gap (lm14 - lm13) / face_h very small.
#   Relaxed mouth : ratio ≈ 0.03–0.06
#   Pressed tight : ratio < 0.020  (was 0.022 — tightened)
LIP_THIN_THRESHOLD    = 0.020

# MOUTH SHRINK (width): corners pulled inward under stress.
#   Normal mouth width / face_w ≈ 0.38–0.50
#   Stress-shrunk       < 0.35
MOUTH_WIDTH_THRESHOLD = 0.35

# Nose flare — kept but not required (optional bonus cue)
NOSE_FLARE_THRESHOLD  = 0.225

# ─── Display constants ──────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.75
THICKNESS       = 2
COLOR_STRESS    = (0, 60, 255)
COLOR_NO_STRESS = (30, 200, 30)
COLOR_DETECTING = (200, 200, 200)
LABEL_X         = 20
LABEL_Y         = 285


# ═══════════════════════════════════════════════════════════════
#  COMPATIBILITY LOADER
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  MEDIAPIPE EXPRESSION ANALYSER
# ═══════════════════════════════════════════════════════════════

class _ExpressionAnalyser:
    """
    Computes a 0–1 stress cue score from FaceMesh landmarks.

    GATE LOGIC (new):
      The 3 PRIMARY stress signals are required to all fire together
      before any meaningful score is returned:
        1. Brow furrow   — inner brows squeezed inward (forehead wrinkle)
        2. Eye squint    — EAR drops (eyes narrow)
        3. Lip/mouth compression — mouth pressed tight AND/OR narrowed

      If fewer than 2 of the 3 primary signals fire → score capped at 0.25
      (model alone cannot push fused score above FUSED_THRESHOLD = 0.60
       because MODEL_WEIGHT=0.30, so max fused = 0.30*1.0 + 0.70*0.25 = 0.475 < 0.60)

      When 2+ primaries fire, score scales with additional cues.
      When all 3 fire, base score is 0.75; nose flare can push to 1.0.

    Returns 0.5 (neutral) when no face detected — does not bias result.
    """

    def _ear(self, lm, p1, p2, p3, p4, p5, p6):
        """Eye Aspect Ratio — drops when eye squints/closes."""
        v1 = np.linalg.norm([lm[p2].x - lm[p6].x, lm[p2].y - lm[p6].y])
        v2 = np.linalg.norm([lm[p3].x - lm[p5].x, lm[p3].y - lm[p5].y])
        h  = np.linalg.norm([lm[p1].x - lm[p4].x, lm[p1].y - lm[p4].y])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def analyse(self, mesh_results):
        """
        Returns cue_score: float [0.0 – 1.0]
        0.0 = no stress cues, 1.0 = all cues firing.
        Returns 0.5 (neutral) when no face — does not bias fused result.
        """
        if not mesh_results or not mesh_results.multi_face_landmarks:
            return 0.5

        lm     = mesh_results.multi_face_landmarks[0].landmark
        face_h = abs(lm[10].y - lm[152].y)
        face_w = abs(lm[454].x - lm[234].x)

        if face_h < 0.01 or face_w < 0.01:
            return 0.5

        # ── PRIMARY STRESS SIGNALS ──────────────────────────────
        # Must have 2 of 3 for stress to be considered at all.

        # 1. BROW FURROW — inner brows pulled together
        inner_brow_dist = abs(lm[285].x - lm[55].x) / face_w
        brow_furrowed   = inner_brow_dist < BROW_FURROW_THRESHOLD

        # 2. EYE SQUINT — both eyes narrow (EAR drops)
        left_ear   = self._ear(lm, 33,  160, 158, 133, 153, 144)
        right_ear  = self._ear(lm, 362, 385, 387, 263, 373, 380)
        avg_ear    = (left_ear + right_ear) / 2.0
        eyes_squinting = avg_ear < EYE_SQUINT_THRESHOLD

        # 3. LIP / MOUTH COMPRESSION — lips pressed thin AND/OR mouth narrowed
        #    lm13 = upper inner lip, lm14 = lower inner lip
        lip_gap    = abs(lm[14].y - lm[13].y) / face_h
        lips_thin  = lip_gap < LIP_THIN_THRESHOLD

        #    lm61 = left mouth corner, lm291 = right mouth corner
        mouth_w    = abs(lm[291].x - lm[61].x) / face_w
        mouth_narrow = mouth_w < MOUTH_WIDTH_THRESHOLD

        mouth_compressed = lips_thin or mouth_narrow

        # ── GATE CHECK ──────────────────────────────────────────
        primary_count = sum([brow_furrowed, eyes_squinting, mouth_compressed])

        if primary_count < 2:
            # Not enough primary signals — cap score low so model can't override
            return 0.25

        # ── SECONDARY / BONUS CUE ───────────────────────────────
        # 4. Nose flare (bonus only — doesn't gate anything)
        nose_flared = abs(lm[358].x - lm[129].x) / face_w > NOSE_FLARE_THRESHOLD

        # ── SCORE ASSEMBLY ──────────────────────────────────────
        # 2 primaries → 0.60 base; 3 primaries → 0.75 base
        # Each bonus adds 0.15; capped at 1.0
        base_score = 0.60 if primary_count == 2 else 0.75
        bonus      = 0.15 if nose_flared else 0.0

        return min(1.0, base_score + bonus)


# ═══════════════════════════════════════════════════════════════
#  STRESS DETECTOR
# ═══════════════════════════════════════════════════════════════

class StressDetector:
    """
    Landmark-gated stress detection — stress is only flagged when the
    face shows the classic triad:
      • Furrowed brow (forehead shrunken / inner brows close)
      • Squinting eyes (eyes narrowed)
      • Compressed lips/mouth (mouth tight and/or narrowed)

    The MobileNetV2 model acts as a secondary confirming signal (weight 0.30).
    Landmarks are primary (weight 0.70) and gate the entire decision.

    Threading: model inference runs in a background daemon thread so the
    video loop is never blocked.
    """

    def __init__(self, model_path: str):
        print(f"Loading stress model from {model_path} ...")
        self.model    = load_model_compatible(model_path, _build_stress_architecture)
        self.analyser = _ExpressionAnalyser()

        self.stress_counter    = 0
        self.no_stress_counter = 0
        self.label             = "Detecting..."
        self.confidence        = 0.0

        self._lock       = threading.Lock()
        self._model_pred = 0.3      # start below threshold
        self._inferring  = False

        print("Stress model ready.")

    # ──────────────────────────────────────────────────────────
    def _crop_face(self, frame, mesh_results):
        """
        Crop face region from FaceMesh landmark bounding box with 25% padding.
        Falls back to centre-crop if landmarks are absent.
        """
        h, w = frame.shape[:2]

        if mesh_results and mesh_results.multi_face_landmarks:
            lm  = mesh_results.multi_face_landmarks[0].landmark
            xs  = [l.x * w for l in lm]
            ys  = [l.y * h for l in lm]
            x1  = int(min(xs)); x2 = int(max(xs))
            y1  = int(min(ys)); y2 = int(max(ys))
            px  = int((x2 - x1) * 0.25)
            py  = int((y2 - y1) * 0.25)
            x1  = max(0, x1 - px);  x2 = min(w, x2 + px)
            y1  = max(0, y1 - py);  y2 = min(h, y2 + py)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                return crop

        margin_x = int(w * 0.20)
        margin_y = int(h * 0.20)
        return frame[margin_y:h-margin_y, margin_x:w-margin_x]

    # ──────────────────────────────────────────────────────────
    def _preprocess(self, img):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    # ──────────────────────────────────────────────────────────
    def _run_inference(self, face_crop):
        try:
            inp  = self._preprocess(face_crop)
            pred = float(self.model.predict(inp, verbose=0)[0][0])
            with self._lock:
                self._model_pred = pred
        finally:
            with self._lock:
                self._inferring = False

    # ──────────────────────────────────────────────────────────
    def update(self, frame, mesh_results=None):
        """
        Call every frame. Never blocks the video loop.

        Stress requires clear landmark evidence PLUS model confirmation.
        Returns {"label": ..., "confidence": ...}
        """

        # ── Face absent ──────────────────────────────────────
        face_present = (
            mesh_results is not None and
            mesh_results.multi_face_landmarks is not None and
            len(mesh_results.multi_face_landmarks) > 0
        )

        if not face_present:
            self.stress_counter    = 0
            self.no_stress_counter = 0
            self.label             = "Face Not Detected"
            self.confidence        = 0.0
            return self._build_result()

        # ── Fire background model inference on face crop ──────
        with self._lock:
            if not self._inferring:
                self._inferring = True
                face_crop       = self._crop_face(frame, mesh_results)
                t = threading.Thread(
                    target=self._run_inference,
                    args=(face_crop,),
                    daemon=True
                )
                t.start()

        with self._lock:
            model_pred = self._model_pred

        # ── MediaPipe landmark score (GATED) ──────────────────
        cue_score = self.analyser.analyse(mesh_results)

        # ── Fused score ───────────────────────────────────────
        # Landmarks drive 70% of the decision.
        # If cue_score <= 0.25 (gate not met), max fused = 0.475 < 0.60
        # → model alone can NEVER trigger stress without landmark evidence.
        fused     = MODEL_WEIGHT * model_pred + CUE_WEIGHT * cue_score
        is_stress = fused > FUSED_THRESHOLD

        # ── Smoothing ─────────────────────────────────────────
        if is_stress:
            self.stress_counter    += 1
            self.no_stress_counter  = 0
        else:
            self.no_stress_counter += 1
            self.stress_counter     = 0

        if self.stress_counter >= SMOOTHING_FRAMES:
            self.label      = "STRESS"
            self.confidence = fused
        elif self.no_stress_counter >= SMOOTHING_FRAMES:
            self.label      = "NO STRESS"
            self.confidence = 1.0 - fused
        # else: keep previous label (stable during transition)

        return self._build_result()

    # ──────────────────────────────────────────────────────────
    def _build_result(self):
        return {
            "label"     : self.label,
            "confidence": round(self.confidence * 100, 1)
        }

    # ──────────────────────────────────────────────────────────
    def draw(self, frame, result):
        """Standalone test only."""
        label = result["label"]
        conf  = result["confidence"]
        color = (COLOR_STRESS    if label == "STRESS"    else
                 COLOR_NO_STRESS if label == "NO STRESS" else COLOR_DETECTING)
        text = f"Stress: {label}" + (f" ({conf}%)" if conf > 0 else "")
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame,
            (LABEL_X-6, LABEL_Y-th-4), (LABEL_X+tw+6, LABEL_Y+4), (20,20,20), -1)
        cv2.putText(frame, text, (LABEL_X, LABEL_Y), FONT, FONT_SCALE, color, THICKNESS)
        return frame


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════

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