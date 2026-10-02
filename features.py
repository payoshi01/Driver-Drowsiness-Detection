"""MediaPipe Face Mesh -> geometric features + eye/mouth crops.
Shared by build_features.py (training) and app.py (runtime) so preprocessing never drifts."""
import cv2, numpy as np
import mediapipe as mp
from config import CROP_SIZE

R_EYE = [33, 160, 158, 133, 153, 144]        # subject's right eye (p1..p6 order for EAR)
L_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_H = [(13, 14), (82, 87), (312, 317)]    # inner-lip vertical pairs
MOUTH_W = (78, 308)                           # inner-lip corners
MOUTH_CROP = [61, 291, 13, 14, 0, 17]
DRAW_IDX = R_EYE + L_EYE + [61, 291, 13, 14]
POSE_IDX = [1, 152, 33, 263, 61, 291]         # nose, chin, eye corners, mouth corners
# generic 3D face model (mm), camera frame: x right, y down, z away from camera
MODEL_3D = np.array([[0, 0, 0], [0, 63.6, 12.5], [-43.3, -32.7, 26], [43.3, -32.7, 26],
                     [-28.9, 28.9, 24.1], [28.9, 28.9, 24.1]], np.float64)

_meshes = {}

def _mesh(static):
    if static not in _meshes:
        _meshes[static] = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return _meshes[static]

def _d(a, b):
    return float(np.linalg.norm(a - b))

def ear(pts, idx):
    p = pts[idx]
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-6)

def mar(pts):
    h = np.mean([_d(pts[a], pts[b]) for a, b in MOUTH_H])
    return h / (_d(pts[MOUTH_W[0]], pts[MOUTH_W[1]]) + 1e-6)

def _wrap(a):
    return (a + 90.0) % 180.0 - 90.0

def head_pose(pts, shape):
    """Returns yaw, pitch, roll in degrees (sign convention arbitrary but consistent)."""
    h, w = shape[:2]
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts[POSE_IDX].astype(np.float64), cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(R)[0]
    return _wrap(yaw), _wrap(pitch), _wrap(roll)

def nose_offset(pts):
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    return (pts[1, 0] - (x0 + x1) / 2) / (x1 - x0 + 1e-6), (pts[1, 1] - (y0 + y1) / 2) / (y1 - y0 + 1e-6)

def _square_crop(img, sel, scale):
    x0, y0 = sel.min(0); x1, y1 = sel.max(0)
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * scale
    h, w = img.shape[:2]
    a, b = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    c, d = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    if c - a < 4 or d - b < 4:
        return np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8)
    return cv2.resize(img[b:d, a:c], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

def extract(frame_bgr, static=False):
    """-> (feats float32[9], eye_rgb 64x64x3 uint8, mouth_rgb 64x64x3 uint8, pts (478,2) float32) or None if no face.
    static=True for independent still images (dataset building), False for video/webcam (tracking)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _mesh(static).process(rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    pts = np.array([[l.x * w, l.y * h] for l in res.multi_face_landmarks[0].landmark], np.float32)
    el, er = ear(pts, L_EYE), ear(pts, R_EYE)
    yaw, pitch, roll = head_pose(pts, frame_bgr.shape)
    dx, dy = nose_offset(pts)
    feats = np.array([el, er, (el + er) / 2, mar(pts), yaw, pitch, roll, dx, dy], np.float32)
    eye = _square_crop(rgb, pts[R_EYE + L_EYE], 1.6)
    mouth = _square_crop(rgb, pts[MOUTH_CROP], 1.6)
    return feats, eye, mouth, pts

def crop_to_array(crop_rgb):
    """uint8 HxWx3 RGB -> float32 3xHxW in [0,1]. The model normalises internally."""
    return crop_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0