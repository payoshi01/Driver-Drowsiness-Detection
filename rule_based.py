import itertools, json, os
import numpy as np
from sklearn.metrics import f1_score
from config import *
from utils import load_split, metrics


def predict_rule(F, th):
    out = np.zeros(len(F), int)
    out[F[:, FEAT["ear_mean"]] < th["ear"]] = LABEL_ID["Drowsy"]
    out[F[:, FEAT["mar"]] > th["mar"]] = LABEL_ID["Yawning"]
    out[(np.abs(F[:, FEAT["yaw"]]) > th["yaw"]) | (np.abs(F[:, FEAT["pitch"]]) > th["pitch"])] = LABEL_ID["LookingAway"]
    return out


if __name__ == "__main__":
    ensure_dirs()
    va = load_split(FEATURES_NPZ, "val")
    grid = dict(ear=np.arange(0.12, 0.31, 0.01), mar=np.arange(0.20, 0.85, 0.05),
                yaw=np.arange(15, 50, 5), pitch=np.arange(10, 40, 5))
    best, best_th = -1, None
    for e, m, y, p in itertools.product(*grid.values()):
        th = dict(ear=float(e), mar=float(m), yaw=float(y), pitch=float(p))
        f1 = f1_score(va["label_id"], predict_rule(va["feats"], th), average="macro", zero_division=0)
        if f1 > best: best, best_th = f1, th
    print("best val macro-F1 %.3f with %s" % (best, best_th))
    json.dump(best_th, open(THRESHOLDS_JSON, "w"), indent=2)

    res = {"thresholds": best_th, "val_macro_f1": best}
    te = load_split(FEATURES_NPZ, "test"); res["test"] = metrics(te["label_id"], predict_rule(te["feats"], best_th))
    if os.path.exists(FEATURES_LOWLIGHT_NPZ):
        for L in LOWLIGHT_LEVELS:
            d = load_split(FEATURES_LOWLIGHT_NPZ, "test", lighting=L)
            if len(d["label_id"]): res[L] = metrics(d["label_id"], predict_rule(d["feats"], best_th))
    json.dump(res, open(os.path.join(RESULTS, "rule_based.json"), "w"), indent=2)
    print({k: (round(v["acc"], 3), round(v["macro_f1"], 3)) for k, v in res.items() if isinstance(v, dict) and "acc" in v})