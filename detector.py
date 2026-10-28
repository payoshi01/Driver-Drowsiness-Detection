"""Drowsiness + distraction detector using MediaPipe Face Mesh landmarks.

Drowsiness  : eyes closed (EAR), yawning (MAR), head nodding down (pitch ratio)
Distraction : head turned left/right (yaw ratio) or down for a sustained time
"""
import time
import cv2
import numpy as np
import mediapipe as mp

# ---- Tunable thresholds (calibrate with your own face/camera) ----
EAR_THRESH = 0.21        # below = eye closed
MAR_THRESH = 0.60        # above = mouth wide open (yawn)
YAW_LOW, YAW_HIGH = 0.30, 0.70   # nose position across face width (0.5 = centered)
PITCH_DOWN = 0.62        # nose-to-chin ratio above = head down
EYES_CLOSED_SEC = 1.2    # sustained closure -> DROWSY
YAWN_SEC = 1.0
DISTRACT_SEC = 2.0

L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _ear(p):
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-6)


class DriverMonitor:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.t = {"eyes": None, "yawn": None, "away": None}
        self.blinks, self.yawns, self.alerts = 0, 0, 0
        self._was_closed = False
        self._was_yawn = False

    def _timer(self, key, active, now):
        if not active:
            self.t[key] = None
            return 0.0
        if self.t[key] is None:
            self.t[key] = now
        return now - self.t[key]

    def process(self, bgr):
        now = time.time()
        h, w = bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            self._timer("eyes", False, now); self._timer("yawn", False, now)
            away = self._timer("away", True, now)   # no face = not watching the road
            status = "DISTRACTED" if away > DISTRACT_SEC else "NO_FACE"
            return self._out(status, ["No face detected"] if status != "OK" else [], {})

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm])

        ear = (_ear(pts[L_EYE]) + _ear(pts[R_EYE])) / 2
        mar = _d(pts[13], pts[14]) / (_d(pts[61], pts[291]) + 1e-6)
        yaw = (pts[1][0] - pts[234][0]) / (pts[454][0] - pts[234][0] + 1e-6)
        eye_mid = (pts[33] + pts[263]) / 2
        pitch = (pts[1][1] - eye_mid[1]) / (pts[152][1] - eye_mid[1] + 1e-6)

        closed = ear < EAR_THRESH
        yawning = mar > MAR_THRESH
        looking_away = yaw < YAW_LOW or yaw > YAW_HIGH or pitch > PITCH_DOWN

        if self._was_closed and not closed:
            self.blinks += 1
        self._was_closed = closed
        if yawning and not self._was_yawn:
            self.yawns += 1
        self._was_yawn = yawning

        t_eyes = self._timer("eyes", closed, now)
        t_yawn = self._timer("yawn", yawning, now)
        t_away = self._timer("away", looking_away, now)

        reasons, status = [], "OK"
        if t_eyes > EYES_CLOSED_SEC:
            reasons.append("Eyes closed too long"); status = "DROWSY"
        if t_yawn > YAWN_SEC:
            reasons.append("Yawning"); status = "DROWSY"
        if t_away > DISTRACT_SEC:
            reasons.append("Looking away from road")
            status = "DISTRACTED" if status == "OK" else "DROWSY+DISTRACTED"

        metrics = {"ear": round(ear, 3), "mar": round(mar, 3),
                   "yaw": round(yaw, 3), "pitch": round(pitch, 3)}
        return self._out(status, reasons, metrics)

    def _out(self, status, reasons, metrics):
        if status not in ("OK", "NO_FACE"):
            self.alerts += 1
        return {"status": status, "reasons": reasons, "metrics": metrics,
                "blinks": self.blinks, "yawns": self.yawns}
