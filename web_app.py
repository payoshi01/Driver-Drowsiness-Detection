"""Web demo (Flask) built on the project's own modules. Desktop version: app.py.
    python web_app.py --ckpt checkpoints/int8/cnn_eager.pt
then open http://127.0.0.1:5000 and press Start Camera (Ctrl+C in the terminal to stop)."""
import argparse, base64, json, os, threading, time
import cv2, numpy as np, torch
from flask import Flask, jsonify, render_template, request
from config import *
from features import extract, crop_to_array, DRAW_IDX
from models import load_model
from app import Smoother          # reuse the Smoother from your desktop app.py (unchanged)

try:                                   # EAR threshold from the rule-based baseline, used for the blink counter
    EAR_THR = float(json.load(open("thresholds.json"))["ear"])
except Exception:
    EAR_THR = 0.26
BLINK_MIN_S, BLINK_MAX_S = 0.05, 1.0   # a closure between these durations counts as one blink


class Session:
    """Per-demo state: temporal smoother plus blink and yawn counters."""
    def __init__(self):
        self.sm = Smoother(); self.blinks = 0; self.yawns = 0
        self.closed_since = None; self.prev_state = None


app = Flask(__name__)
LOCK = threading.Lock()          # one model and one session: single local user
MODEL = None
S = Session()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    global S
    with LOCK:
        S = Session()
    return jsonify({"ok": True})


def describe(face, state, alarm, perclos, ear, mar, yaw):
    """Map the smoothed state to the status and reasons shown by the page."""
    if alarm:
        return "DROWSY", [f"Sustained drowsiness (PERCLOS {perclos:.2f})"]
    if state == LABEL_ID["Drowsy"]:
        return "DROWSY", [f"Drowsy frames detected (PERCLOS {perclos:.2f})"]
    if state == LABEL_ID["Yawning"]:
        return "YAWNING", [f"Mouth open (MAR {mar:.2f})"] if mar is not None else ["Yawning"]
    if state == LABEL_ID["LookingAway"]:
        return "LOOKING_AWAY", [f"Head turned away (yaw {yaw:.0f} deg)"] if face else ["No face for several frames"]
    return ("OK", []) if face else ("NO_FACE", [])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)["image"].split(",", 1)[-1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    except (KeyError, TypeError, AttributeError, ValueError):
        return jsonify({"error": "bad request"}), 400
    if frame is None:
        return jsonify({"error": "bad frame"}), 400
    h, w = frame.shape[:2]
    with LOCK:
        now = time.time(); out, pred = extract(frame), None
        face, ear, mar, yaw, pitch, lm = False, None, None, None, None, []
        if out is not None:
            feats, eye, mouth, pts = out
            with torch.no_grad():
                logits = MODEL(torch.from_numpy(crop_to_array(eye))[None], torch.from_numpy(crop_to_array(mouth))[None],
                               torch.from_numpy(feats)[None])
            pred = int(logits.argmax(1)); face = True
            ear, mar = float(feats[FEAT["ear_mean"]]), float(feats[FEAT["mar"]])
            yaw, pitch = float(feats[FEAT["yaw"]]), float(feats[FEAT["pitch"]])
            lm = [[float(pts[i][0]) / w, float(pts[i][1]) / h] for i in DRAW_IDX]
            if ear < EAR_THR:                                   # blink counter (time-based, frame-rate independent)
                if S.closed_since is None: S.closed_since = now
            elif S.closed_since is not None:
                if BLINK_MIN_S <= now - S.closed_since <= BLINK_MAX_S: S.blinks += 1
                S.closed_since = None
        else:
            S.closed_since = None
        state, alarm, perclos = S.sm.update(pred, now)
        if state == LABEL_ID["Yawning"] and S.prev_state != LABEL_ID["Yawning"]: S.yawns += 1
        S.prev_state = state
        status, reasons = describe(face, state, alarm, perclos, ear, mar, yaw)
        r = lambda v, n: None if v is None else round(v, n)
        res = {"status": status, "reasons": reasons, "blinks": S.blinks, "yawns": S.yawns,
               "metrics": {"ear": r(ear, 2), "mar": r(mar, 2), "yaw": r(yaw, 0), "pitch": r(pitch, 0)},
               "face": face, "state": CLASSES[state], "state_id": int(state), "alarm": bool(alarm),
               "perclos": round(float(perclos), 3), "landmarks": lm}
    return jsonify(res)


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CKPT_FP32, "cnn.pt"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    MODEL = load_model(a.ckpt)
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
