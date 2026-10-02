import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, pandas as pd
from tqdm import tqdm
from config import *

def darken(img, b, g, sigma, rng):
    x=np.power(img.astype(np.float32)/255.0, g)*b
    x+=rng.normal(0, sigma/255.0, x.shape)
    return (np.clip(x, 0, 1)*255).astype(np.uint8)

df=pd.read_csv(LABELS_CSV)
test=df[(df.split=="test") & (df.source!="mrl")]
rng=np.random.RandomState(SEED); out=[]
for lvl, (b, g, s) in LOWLIGHT_LEVELS.items():
    os.makedirs(os.path.join(LOWLIGHT_DIR, lvl), exist_ok=True)
    for _, r in tqdm(test.iterrows(), total=len(test), desc=lvl):
        img=cv2.imread(os.path.join(DATASET, r.filepath))
        if img is None: continue
        dst=os.path.join("lowlight", lvl, os.path.basename(r.filepath))
        cv2.imwrite(os.path.join(DATASET, dst), darken(img, b, g, s, rng))
        d=r.to_dict(); d["orig_filepath"]=r.filepath; d["filepath"]=dst; d["lighting"]=lvl; out.append(d)
pd.DataFrame(out).to_csv(LABELS_LOWLIGHT_CSV, index=False)
print("wrote", LABELS_LOWLIGHT_CSV, len(out), "rows")