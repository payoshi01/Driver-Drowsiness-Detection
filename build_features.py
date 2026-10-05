import argparse, os
import cv2, numpy as np, pandas as pd
from tqdm import tqdm
from config import *
from features import extract

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=LABELS_CSV)
ap.add_argument("--out", default=FEATURES_NPZ)
ap.add_argument("--ref", default=None, help="clean features.npz whose labels are copied (for low-light sets)")
a = ap.parse_args()
ensure_dirs()

df = pd.read_csv(a.labels)
df = df[df.source.isin(["yawdd", "own"])].reset_index(drop=True)
N = len(df)
F = np.zeros((N, NUM_FEATURES), np.float32)
E = np.zeros((N, CROP_SIZE, CROP_SIZE, 3), np.uint8); M = np.zeros_like(E)
found = np.zeros(N, np.uint8)
for i, r in enumerate(tqdm(df.itertuples(), total=N)):
    img = cv2.imread(os.path.join(DATASET, r.filepath))
    out = extract(img, static=True) if img is not None else None
    if out is not None:
        F[i], E[i], M[i], _ = out; found[i] = 1
y = df.label_id.values.astype(np.int64).copy()

if a.ref:                                    
    ref = np.load(a.ref)
    lab = dict(zip(ref["filepath"], ref["label_id"]))
    y = np.array([lab.get(p, -1) for p in df.orig_filepath], np.int64)
else:                                        
    m = (y == LABEL_ID["Yawning"]) & (found == 1) & (F[:, FEAT["mar"]] < YAWN_MAR_MIN)
    y[m] = LABEL_ID["Focused"]
    print(f"relabelled {m.sum()} closed-mouth frames in yawning clips -> Focused (spot-check these!)")

print(f"faces found in {found.sum()}/{N} frames")
S = lambda col: np.array(df[col].fillna("").tolist(), dtype=str)   
np.savez(a.out, feats=F, eye=E, mouth=M, face_found=found, label_id=y, split=S("split"), source=S("source"),
         subject=S("subject_id"), lighting=S("lighting"), filepath=S("filepath"))
print("saved", a.out)