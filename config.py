"""Single source of truth for paths, class order, constants and app settings."""
import os, platform, random
import numpy as np

SEED = 42
CLASSES = ["Focused", "Drowsy", "Yawning", "LookingAway"]     # ids 0..3 -- never reorder
LABEL_ID = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ---- paths ----
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT, "dataset")
RAW = os.path.join(DATASET, "raw")
FRAMES = os.path.join(DATASET, "frames")
LOWLIGHT_DIR = os.path.join(DATASET, "lowlight")
LABELS_CSV = os.path.join(DATASET, "labels.csv")
LABELS_LOWLIGHT_CSV = os.path.join(DATASET, "labels_lowlight.csv")
DATA_DIR = os.path.join(ROOT, "data")
FEATURES_NPZ = os.path.join(DATA_DIR, "features.npz")
FEATURES_LOWLIGHT_NPZ = os.path.join(DATA_DIR, "features_lowlight.npz")
CKPT_FP32 = os.path.join(ROOT, "checkpoints", "fp32")
CKPT_INT8 = os.path.join(ROOT, "checkpoints", "int8")
RESULTS = os.path.join(ROOT, "results")
THRESHOLDS_JSON = os.path.join(ROOT, "thresholds.json")
ALARM_WAV = os.path.join(ROOT, "alarm.wav")

# ---- data ----
CROP_SIZE = 64
FEATURE_NAMES = ["ear_l", "ear_r", "ear_mean", "mar", "yaw", "pitch", "roll", "nose_dx", "nose_dy"]
NUM_FEATURES = len(FEATURE_NAMES)
FEAT = {n: i for i, n in enumerate(FEATURE_NAMES)}
SPLIT_FRACS = (0.70, 0.15, 0.15)   # train, val, test (by person)
RECORD_SECONDS = 30
FRAME_STEP = 5            # keep every 5th frame (~6 fps)
TRIM_FRAMES = 30          # drop first/last second of own clips (transitions)
YAWN_MAR_MIN = 0.35       # frames in "yawning" clips with MAR below this are relabelled Focused
LOWLIGHT_LEVELS = {"L1": (0.60, 1.5, 3), "L2": (0.35, 2.0, 8), "L3": (0.15, 2.5, 15)}  # brightness, gamma, noise sigma

# ---- app / temporal logic ----
SMOOTH_FRAMES = 9         # majority vote for the displayed state
ALERT_FRAMES = 45         # consecutive Drowsy frames before alarm (~1.5 s at 30 fps)
RELEASE_FRAMES = 15       # non-drowsy frames needed to reset the run
NOFACE_FRAMES = 15        # no face this long -> LookingAway
PERCLOS_WINDOW_S = 30
PERCLOS_THRESH = 0.20

# ---- quantization engine: qnnpack on ARM (Raspberry Pi, Apple silicon), fbgemm on x86 ----
QENGINE = "qnnpack" if platform.machine().lower() in ("aarch64", "arm64", "armv7l") else "fbgemm"

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def ensure_dirs():
    for d in (DATA_DIR, CKPT_FP32, CKPT_INT8, RESULTS, FRAMES):
        os.makedirs(d, exist_ok=True)