"""Real-time demo.  python app.py --ckpt checkpoints/int8/fusion_eager.pt   (q to quit)"""
import argparse, os, time
from collections import Counter, deque
import cv2, numpy as np, torch
from config import *
from features import extract, crop_to_array, DRAW_IDX
from models import load_model


class Smoother:
    """Majority-vote state + alarm (N consecutive Drowsy frames OR PERCLOS window) with hysteresis."""
    def __init__(self):
        self.hist = deque(maxlen=SMOOTH_FRAMES); self.state = 0; self.noface = 0
        self.run = 0; self.clear = 0; self.alarm = False; self.win = deque()

    def update(self, pred, t):
        if pred is None:                                   # no face: hold state, then LookingAway
            self.noface += 1
            pred = LABEL_ID["LookingAway"] if self.noface >= NOFACE_FRAMES else self.state
        else:
            self.noface = 0
        closed = pred == LABEL_ID["Drowsy"]
        self.win.append((t, closed))
        while self.win and t - self.win[0][0] > PERCLOS_WINDOW_S: self.win.popleft()
        self.hist.append(pred); self.state = Counter(self.hist).most_common(1)[0][0]
        if closed: self.run += 1; self.clear = 0
        else:
            self.clear += 1
            if self.clear >= RELEASE_FRAMES: self.run = 0
        perclos = sum(c for _, c in self.win) / max(len(self.win), 1)
        span = t - self.win[0][0]
        if self.run >= ALERT_FRAMES or (perclos > PERCLOS_THRESH and span > 10): self.alarm = True
        elif self.run == 0 and perclos < 0.8 * PERCLOS_THRESH: self.alarm = False
        return self.state, self.alarm, perclos


class Alarm:
    def __init__(self, enabled):
        self.ok = False
        if enabled:
            try:
                import pygame; pygame.mixer.init(); self.pg = pygame; self.snd = pygame.mixer.Sound(ALARM_WAV); self.ok = True
            except Exception as e:
                print("alarm disabled:", e, "(run python data_tools/make_alarm.py, pip install pygame)")

    def play(self):
        if self.ok and not self.pg.mixer.get_busy(): self.snd.play()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CKPT_FP32, "fusion.pt")); ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--no-alarm", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, sm, alarm = load_model(a.ckpt), Smoother(), Alarm(not a.no_alarm)
    cap = cv2.VideoCapture(a.camera); cap.set(3, 640); cap.set(4, 480)
    fps, t_prev = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        now = time.time(); out = extract(frame); pred = None
        if out is not None:
            feats, eye, mouth, pts = out
            with torch.no_grad():
                logits = model(torch.from_numpy(crop_to_array(eye))[None], torch.from_numpy(crop_to_array(mouth))[None],
                               torch.from_numpy(feats)[None])
            pred = int(logits.argmax(1))
            for i in DRAW_IDX: cv2.circle(frame, tuple(int(v) for v in pts[i]), 1, (0, 255, 0), -1)
            cv2.putText(frame, f"EAR {feats[FEAT['ear_mean']]:.2f} MAR {feats[FEAT['mar']]:.2f} yaw {feats[FEAT['yaw']]:.0f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        state, on, perclos = sm.update(pred, now)
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6); t_prev = now
        cv2.putText(frame, f"{CLASSES[state]}  PERCLOS {perclos:.2f}  FPS {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if on:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)
            cv2.putText(frame, "DROWSINESS ALERT", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3); alarm.play()
        cv2.imshow("driver monitor", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("r"): sm=Smoother()
        if k == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()