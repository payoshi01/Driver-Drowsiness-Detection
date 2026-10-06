import argparse, copy, json, os
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from models import build_model, SmallCNN
from utils import load_split, batches, predict, metrics

EYE_BACKBONE = os.path.join(CKPT_FP32, "eye_backbone.pt")


def pretrain_eye(epochs=3, n_train=30000, n_val=5000):
    df = pd.read_csv(LABELS_CSV); df = df[df.source == "mrl"]

    def load(sub):
        X, y = [], []
        for p, s in zip(sub.filepath, sub.eye_state):
            im = cv2.imread(os.path.join(DATASET, p), 0)
            if im is not None:
                X.append(cv2.resize(im, (64, 64))); y.append(int(s == "open"))
        X = torch.from_numpy(np.stack(X)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
        return (X - 0.5) / 0.5, torch.tensor(y)

    tr = df[df.split == "train"]; va = df[df.split == "val"]
    Xt, yt = load(tr.sample(min(len(tr), n_train), random_state=SEED))
    Xv, yv = load(va.sample(min(len(va), n_val), random_state=SEED))
    net, head = SmallCNN(), nn.Linear(64, 2)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), 1e-3)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            j = perm[i:i + 256]
            loss = nn.functional.cross_entropy(head(net(Xt[j])), yt[j]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): acc = (head(net(Xv)).argmax(1) == yv).float().mean().item()
        print(f"[MRL pretrain] epoch {ep+1} val acc {acc:.3f}")
    torch.save(net.state_dict(), EYE_BACKBONE); print("saved", EYE_BACKBONE)


def train_one(name, a):
    seed_everything(); ensure_dirs()
    tr, va, te = (load_split(FEATURES_NPZ, s) for s in ("train", "val", "test"))
    model = build_model(name, pretrained=a.pretrained)
    model.f_mean.copy_(torch.from_numpy(tr["feats"].mean(0)))
    model.f_std.copy_(torch.from_numpy(tr["feats"].std(0) + 1e-6))
    if model.use_crops and model.backbone == "small" and os.path.exists(EYE_BACKBONE) and not a.no_pretrained_eye:
        model.enc_eye.load_state_dict(torch.load(EYE_BACKBONE)); print("init enc_eye from MRL")
    cnt = np.bincount(tr["label_id"], minlength=NUM_CLASSES).astype(np.float32)
    w = torch.tensor(cnt.sum() / (NUM_CLASSES * np.maximum(cnt, 1)))
    print(name, "train class counts", cnt.astype(int).tolist())
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train(); tot = 0
        for eye, mouth, feats, y in batches(tr, a.bs, shuffle=True, augment=True, seed=ep):
            loss = crit(model(eye, mouth, feats), y); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        f1 = metrics(va["label_id"], predict(model, va))["macro_f1"]
        print(f"{name} ep {ep+1:02d} loss {tot:.2f} val macro-F1 {f1:.3f}")
        if f1 > best: best, best_state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(dict(name=name, state_dict=best_state, classes=CLASSES, val_macro_f1=best), os.path.join(CKPT_FP32, f"{name}.pt"))
    res = metrics(te["label_id"], predict(model, te))
    json.dump(res, open(os.path.join(RESULTS, f"ablation_{name}.json"), "w"), indent=2)
    cm = np.array(res["cm"]); fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES); ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(name)
    for i in range(4):
        for j in range(4): ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"cm_{name}.png"), dpi=200); plt.close(fig)
    print(f"== {name}: TEST acc {res['acc']:.3f} macro-F1 {res['macro_f1']:.3f}")
    return dict(model=name, acc=res["acc"], macro_f1=res["macro_f1"], **{f"f1_{c}": f for c, f in zip(CLASSES, res["f1_per_class"])})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["geo", "cnn", "fusion"])
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet weights for MobileNetV3")
    ap.add_argument("--pretrain-eye", action="store_true"); ap.add_argument("--no-pretrained-eye", action="store_true")
    a = ap.parse_args(); ensure_dirs()
    if a.pretrain_eye:
        pretrain_eye()
    else:
        rows = [train_one(m, a) for m in a.models]
        out = os.path.join(RESULTS, "ablation.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(out):
            old = pd.read_csv(out); new = pd.concat([old[~old.model.isin(new.model)], new])
        new.to_csv(out, index=False); print(new.to_string(index=False))