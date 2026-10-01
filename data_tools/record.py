import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from config import RAW, RECORD_SECONDS
if len(sys.argv)!=4:
    sys.exit("usage: python data_tools.record.py <person_id> <bright|dim|dark> <glasses|noglasses>")
person, light, glasses=sys.argv[1:4]
KEYS={ord("1"): "focused", ord("2"): "drowsy", ord("3"): "yawning", ord("4"): "lookaway"}
HINT={"focused": "look at screen, blink normally", "drowsy": "slow blinks, eyes closed 2-3 s",
        "yawning": "yawn, mouth wide 2-4 s", "lookaway": "turn head / look at phone"}
out_dir=os.path.join(RAW, "own", person); os.makedirs(out_dir, exist_ok=True)

cap=cv2.VideoCapture(0); cap.set(3, 640); cap.set(4, 480)
writer, cur, t0=None, None, 0
while True:
    ok, frame=cap.read()
    if not ok: break
    disp=frame.copy()
    if writer:
        writer.write(frame)
        left=RECORD_SECONDS-(time.time()-t0)
        cv2.putText(disp, f"REC {cur} {left:0.0f}s-{HINT[cur]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if left<=0: writer.release(); writer=None
    else:
        cv2.putText(disp, "1 focused 2 drowsy 3 yawning 4 lookaway | q quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imshow("record", disp)
    k=cv2.waitKey(1) & 0xFF
    if k in KEYS and writer is None:
        cur=KEYS[k]; t0=time.time()
        fn=os.path.join(out_dir, f"{person}_{cur}_{light}_{glasses}_{int(t0)}.mp4")
        writer=cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    elif k==ord("s") and writer:
        writer.release(); writer=None
    elif k==ord("q"):
        break
if writer:writer.release()
cap.release(); cv2.destroyAllWindows()
