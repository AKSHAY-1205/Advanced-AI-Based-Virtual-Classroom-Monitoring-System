import cv2
import mediapipe as mp

# ================= CONSTANTS =================

VISIBILITY_THRESHOLD = 0.5
PALM_RAISE_THRESHOLD = 10    # wrist above elbow by this many pixels = palm raise
ARM_RAISE_THRESHOLD  = 20    # wrist above shoulder by this many pixels = arm raise

FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.85
THICKNESS    = 2
COLOR_RAISED = (0, 255, 0)
COLOR_NO     = (30, 200, 30)
LABEL_X      = 20
LABEL_Y      = 240


# ================= HAND RAISE DETECTOR =================

class HandRaiseDetector:
    """
    Detects two levels of hand raise:
      - Palm Raise : wrist above elbow (just lifting palm/forearm)
      - Arm Raise  : wrist above shoulder (full arm raise)

    Result keys: hand_raised (bool), raise_type (str), hand_side (str)

    Frame is flipped (mirror), so MediaPipe LEFT = screen RIGHT.
    Labels are corrected to match what user sees on screen.
    """

    def __init__(self):
        self.hand_raised = False
        self.raise_type  = None
        self.hand_side   = None

    # --------------------------------------------------
    def update(self, pose_results, frame_height):
        """
        Args:
            pose_results : mp Pose result
            frame_height : frame height in pixels

        Returns:
            dict — hand_raised, raise_type, hand_side
        """
        self.hand_raised = False
        self.raise_type  = None
        self.hand_side   = None

        if not pose_results or not pose_results.pose_landmarks:
            return self._build_result()

        lm = pose_results.pose_landmarks.landmark
        h  = frame_height

        def py(lmk):
            return int(lmk.y * h)

        def vis(lmk):
            return lmk.visibility > VISIBILITY_THRESHOLD

        # After cv2.flip(frame,1): MediaPipe LEFT → appears on screen RIGHT
        checks = [
            (lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST],
             lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW],
             lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
             "Right"),    # screen right (because frame is mirrored)

            (lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
             lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW],
             lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER],
             "Left"),     # screen left
        ]

        best_type = None
        best_side = None

        for wrist, elbow, shoulder, side in checks:
            if not (vis(wrist) and vis(elbow)):
                continue

            wrist_y    = py(wrist)
            elbow_y    = py(elbow)
            shoulder_y = py(shoulder)

            arm_raised  = vis(shoulder) and (wrist_y < shoulder_y - ARM_RAISE_THRESHOLD)
            palm_raised = wrist_y < elbow_y - PALM_RAISE_THRESHOLD

            if arm_raised:
                if best_type != "Arm Raise":
                    best_type = "Arm Raise"
                    best_side = side
                elif best_side != side:
                    best_side = "Both"

            elif palm_raised:
                if best_type is None:
                    best_type = "Palm Raise"
                    best_side = side
                elif best_type == "Palm Raise" and best_side != side:
                    best_side = "Both"

        if best_type:
            self.hand_raised = True
            self.raise_type  = best_type
            self.hand_side   = best_side

        return self._build_result()

    # --------------------------------------------------
    def _build_result(self):
        return {
            "hand_raised": self.hand_raised,
            "raise_type" : self.raise_type,   # "Palm Raise" / "Arm Raise" / None
            "hand_side"  : self.hand_side      # "Left" / "Right" / "Both" / None
        }

    # --------------------------------------------------
    def draw(self, frame, result):
        if result["hand_raised"]:
            text  = f"Hand Raised: {result['raise_type']} ({result['hand_side']})"
            color = COLOR_RAISED
        else:
            text  = "Hand Raised: NO"
            color = COLOR_NO

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
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    detector = HandRaiseDetector()
    cap      = cv2.VideoCapture(0)

    print("Hand Raise Detection — Press Q to quit")
    print("  Palm Raise = wrist above elbow")
    print("  Arm Raise  = wrist above shoulder")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame        = cv2.flip(frame, 1)
        h, w, _      = frame.shape
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result  = pose.process(rgb)

        result = detector.update(pose_result, h)
        frame  = detector.draw(frame, result)

        cv2.imshow("Hand Raise Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()