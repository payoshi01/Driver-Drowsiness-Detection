import argparse
import os
import torch
from torch.ao.quantization import get_default_qconfig, get_default_qconfig_mapping, prepare, convert
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from config import *
from models import load_model
from utils import load_split, batches, predict, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mode", choices=["eager", "fx"], required=True)
ap.add_argument("--calib", type=int, default=400)
a = ap.parse_args()
seed_everything()
ensure_dirs()
torch.backends.quantized.engine = QENGINE

src = os.path.join(CKPT_FP32, f"{a.model}.pt")
model = load_model(src)
assert model.use_crops, "geometry-only model has no image encoder to quantize"
tr = load_split(FEATURES_NPZ, "train")
idx = torch.randperm(len(tr["label_id"]))[:a.calib].numpy()
calib = {k: v[idx] for k, v in tr.items()}

names = ["enc_eye", "enc_mouth"]
if a.mode == "eager":
    assert model.backbone == "small", "use --mode fx for MobileNetV3"
    for n in names:
        enc = getattr(model, n)
        enc.fuse()
        enc.qconfig = get_default_qconfig(QENGINE)
        prepare(enc, inplace=True)
else:
    qmap = get_default_qconfig_mapping(QENGINE)
    for n in names:
        setattr(model, n, prepare_fx(getattr(model, n).eval(),
                qmap, (torch.randn(1, 3, 64, 64),)))

with torch.no_grad():                                   # calibration pass (half the samples are darkened)
    for eye, mouth, feats, _ in batches(calib, 50, augment=True):
        model(eye, mouth, feats)

for n in names:
    if a.mode == "eager":
        convert(getattr(model, n), inplace=True)
    else:
        setattr(model, n, convert_fx(getattr(model, n)))

out = os.path.join(CKPT_INT8, f"{a.model}_{a.mode}.pt")
# full-model pickle (load with weights_only=False)
torch.save(dict(name=a.model, mode=a.mode, quant=True, state_dict=model.state_dict()), out)
va = load_split(FEATURES_NPZ, "val")
m = metrics(va["label_id"], predict(model, va))
print(f"saved {out}  ({os.path.getsize(out)/1e6:.2f} MB, FP32 was {os.path.getsize(src)/1e6:.2f} MB)")
print(f"INT8 val acc {m['acc']:.3f} macro-F1 {m['macro_f1']:.3f}")
