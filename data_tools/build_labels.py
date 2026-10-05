import os, glob, re, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from config import *

rows=[]
rel=lambda p: os.path.relpath(p, DATASET)
for p in glob.glob(os.path.join(RAW, "mrl", "**", "*.png"), recursive=True):
    parts=os.path.basename(p)[:-4].split("_")
    if len(parts)!=8: continue
    s, _, gender, glasses, eye, refl, light, sensor=parts
    rows.append(dict(filepath=rel(p), source="mrl", subject_id=f"mrl_{s}", clip_id="", label="NA", label_id=-1,
                     eye_state="open" if eye=="1" else "closed", lighting="bright" if light=="1" else "dim", 
                     glasses=int(glasses)))

NAME={"focused": "Focused", "drowsy": "Drowsy", "yawning": "Yawning", "lookaway": "LookingAway"}
for p in glob.glob(os.path.join(FRAMES, "own", "*.jpg")):
    b=os.path.basename(p)[:-4].split("_")
    if len(b)!=6 or b[1] not in NAME: continue
    person, cls, light, gl=b[:4]; lab=NAME[cls]
    rows.append(dict(filepath=rel(p), source="own", subject_id=f"own_{person}", clip_id="_".join(b[:5]), label=lab, 
                     label_id=LABEL_ID[lab], eye_state="closed" if lab=="Drowsy" else ("open" if lab=="Focused" else ""),
                     lighting=light, glasses=int(gl=="glasses")))

for p in glob.glob(os.path.join(FRAMES, "yawdd", "*.jpg")):
    stem=os.path.basename(p)
    m=re.match(r"(\d+)-(Male|Female)(NoGlasses|Glasses|Sunglasses)?", stem, re.I)
    if not m: continue
    low=stem.lower()
    lab="Yawning" if "yawn" in low else ("Focused" if ("normal" in low or "talk" in low) else None)
    if lab is None: continue
    rows.append(dict(filepath=rel(p), source="yawdd", subject_id=f"yaw_{m.group(2)[0].upper()}{m.group(1)}",
                     clip_id=stem.rsplit("_f", 1)[0], label=lab, label_id=LABEL_ID[lab], eye_state="", lighting="",
                     glasses=int(bool(m.group(3)) and m.group(3).lower()!="noglasses")))

df=pd.DataFrame(rows)
if df.empty: sys.exit("No files found. Did you download the datasets / extract frames?")

rng=np.random.RandomState(SEED)
split_of={}
for src, g in df.groupby("source"):
    subs=sorted(g.subject_id.unique()); rng.shuffle(subs); n=len(subs)
    n_test=max(1, round(n*SPLIT_FRACS[2])); n_val=max(1, round(n*SPLIT_FRACS[1])) if n>=3 else 0
    for i, s in enumerate(subs):
        split_of[s]="test" if i<n_test else("val" if i<n_test+n_val else "train")
df["split"]=df.subject_id.map(split_of)
df.to_csv(LABELS_CSV, index=False)
if "--freeze" in sys.argv:
    shutil.copy(LABELS_CSV, os.path.join(DATASET, "labels_v1.csv"))
    print("froze labels_v1.csv") 

S={k: set(v.subject_id) for k, v in df.groupby("split")}
e=set()
assert not (S.get("train", e) & S.get("val", e)) and not (S.get("train", e) & S.get("test", e)) and not (S.get("val", e) & S.get("test", e)), "LEAKAGE: subject in two splits"
print(df.groupby(["source", "split"]).subject_id.nunique().unstack(fill_value=0))
print(df[df.label!="NA"].groupby(["split", "label"]).size().unstack(fill_value=0))
miss=set(CLASSES)-set(df[(df.split=="test") & (df.label!="NA")].label)
if miss: print("WARNING: classes missing from test split:", miss, "-> record more people.change SEED")   