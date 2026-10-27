import argparse
import os
import time
import numpy as np
import pandas as pd
import psutil
import torch
from config import *
from models import load_model
from utils import load_split, metrics, predict, batches


def bench(a):
    torch.set_num_threads(a.threads)
    model = load_model(a.ckpt)
    row = dict(name=a.name, precision=a.precision, engine=QENGINE, threads=a.threads,
               size_mb=round(os.path.getsize(a.ckpt) / 1e6, 3))
    te = load_split(FEATURES_NPZ, "test")
    m = metrics(te["label_id"], predict(model, te))
    row.update(acc=m["acc"], macro_f1=m["macro_f1"])
    if os.path.exists(FEATURES_LOWLIGHT_NPZ):
        for L in LOWLIGHT_LEVELS:
            d = load_split(FEATURES_LOWLIGHT_NPZ, "test", lighting=L)
            if len(d["label_id"]):
                mm = metrics(d["label_id"], predict(model, d))
                row[f"acc_{L}"] = mm["acc"]
                row[f"f1_{L}"] = mm["macro_f1"]
    # ---- model-only latency, batch 1 ----
    small = {k: v[:50] for k, v in te.items()}
    inp = [(e, mo, f) for e, mo, f, _ in batches(small, 1)]
    proc = psutil.Process()
    proc.cpu_percent(None)
    with torch.no_grad():
        for i in range(a.warmup):
            model(*inp[i % len(inp)])
        t = []
        for i in range(a.runs):
            t0 = time.perf_counter()
            model(*inp[i % len(inp)])
            t.append((time.perf_counter() - t0) * 1000)
    row.update(cpu_pct=proc.cpu_percent(None), rss_mb=round(proc.memory_info().rss / 1e6, 1),
               lat_ms_mean=round(float(np.mean(t)), 3), lat_ms_p95=round(float(np.percentile(t, 95)), 3),
               model_fps=round(1000 / float(np.mean(t)), 1))
    # ---- end-to-end (frame -> Face Mesh -> model), optional ----
    if a.e2e_video:
        import cv2
        from features import extract, crop_to_array
        cap = cv2.VideoCapture(a.e2e_video)
        t = []
        while len(t) < 300:
            ok, fr = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            out = extract(fr)
            if out is not None:
                f, e, mo, _ = out
                with torch.no_grad():
                    model(torch.from_numpy(crop_to_array(e))[None], torch.from_numpy(
                        crop_to_array(mo))[None], torch.from_numpy(f)[None])
            t.append((time.perf_counter() - t0) * 1000)
        row.update(e2e_ms_mean=round(float(np.mean(t)), 2),
                   e2e_fps=round(1000 / float(np.mean(t)), 1))
    out = os.path.join(RESULTS, "benchmark.csv")
    df = pd.DataFrame([row])
    if os.path.exists(out):
        old = pd.read_csv(out)
        df = pd.concat(
            [old[~((old.name == a.name) & (old.precision == a.precision))], df])
    df.to_csv(out, index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(os.path.join(RESULTS, "benchmark.csv"))
    os.makedirs(os.path.join(RESULTS, "charts"), exist_ok=True)
    base = df[df.precision == "fp32"].set_index("name")
    df["acc_drop"] = [base.acc[n] -
                      a if n in base.index else np.nan for n, a in zip(df.name, df.acc)]
    if "acc_L3" in df:
        df["acc_drop_L3"] = [base.acc_L3[n] -
                             a if n in base.index else np.nan for n, a in zip(df.name, df.acc_L3)]
    df.to_csv(os.path.join(RESULTS, "benchmark_summary.csv"), index=False)
    df["label"] = df.name + "\n" + df.precision
    for col, title in [("lat_ms_mean", "Model latency (ms, batch 1)"), ("model_fps", "Model FPS"), ("size_mb", "Model size (MB)"),
                       ("acc", "Test accuracy"), ("acc_L3", "Accuracy, low-light L3"), ("cpu_pct", "CPU %"), ("rss_mb", "RSS (MB)")]:
        if col not in df:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(df.label, df[col])
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "charts", f"{col}.png"), dpi=200)
        plt.close(fig)
    print(df.drop(columns="label").to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--precision")
    ap.add_argument("--ckpt")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--e2e-video", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    ensure_dirs()
    plot() if a.plot else bench(a)
