import numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from config import NUM_CLASSES


def load_split(npz_path, split, lighting=None, sources=("yawdd", "own")):
    d = np.load(npz_path)
    m = (d["split"] == split) & (d["face_found"] == 1) & (d["label_id"] >= 0) & np.isin(d["source"], sources)
    if lighting:
        m &= d["lighting"] == lighting
    return {k: d[k][m] for k in ("eye", "mouth", "feats", "label_id", "subject")}


def augment_light(x):
    
    B = x.size(0)
    on = (torch.rand(B, 1, 1, 1) < 0.5).float()
    g = 1 + torch.rand(B, 1, 1, 1) * 1.2
    b = 0.3 + torch.rand(B, 1, 1, 1) * 0.7
    dark = x.clamp(0, 1) ** g * b + torch.randn_like(x) * torch.rand(B, 1, 1, 1) * 0.06
    return (on * dark + (1 - on) * x).clamp(0, 1)


def batches(d, bs=128, shuffle=False, augment=False, seed=0):
    n = len(d["label_id"]); idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        eye = torch.from_numpy(d["eye"][j]).permute(0, 3, 1, 2).float() / 255
        mouth = torch.from_numpy(d["mouth"][j]).permute(0, 3, 1, 2).float() / 255
        if augment:
            eye, mouth = augment_light(eye), augment_light(mouth)
        yield eye, mouth, torch.from_numpy(d["feats"][j]).float(), torch.from_numpy(d["label_id"][j]).long()


@torch.no_grad()
def predict(model, d, bs=256):
    model.eval()
    p = [model(e, m, f).argmax(1).numpy() for e, m, f, _ in batches(d, bs)]
    return np.concatenate(p) if p else np.array([], dtype=int)


def metrics(y, p):
    if len(y) == 0:
        raise ValueError("empty evaluation set - check splits / face detection")
    L = list(range(NUM_CLASSES))
    return dict(acc=float(accuracy_score(y, p)),
                macro_f1=float(f1_score(y, p, labels=L, average="macro", zero_division=0)),
                f1_per_class=f1_score(y, p, labels=L, average=None, zero_division=0).tolist(),
                cm=confusion_matrix(y, p, labels=L).tolist())