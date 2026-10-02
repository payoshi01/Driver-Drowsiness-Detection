import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import cv2
from tqdm import tqdm
from config import RAW, FRAMES, FRAME_STEP, TRIM_FRAMES

def extract(video, out_dir, trim):
    os.makedirs(out_dir, exist_ok=True)
    cap=cv2.VideoCapture(video); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); stem=Path(video).stem
    for i in range(n):
        ok, f=cap.read()
        if not ok: break
        if i%FRAME_STEP==0 and trim<=i<n-trim:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_f{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()

if __name__=="__main__":
    own_videos=sorted(glob.glob(os.path.join(RAW, "own", "*", "*.mp4")))
    for video in tqdm(own_videos, desc="own"):
        extract(video, os.path.join(FRAMES, "own"), TRIM_FRAMES)
    yawdd_videos=sorted(glob.glob(os.path.join(RAW, "yawdd", "**", "*.avi"), recursive=True))
    for video in tqdm(yawdd_videos, desc="yawdd"):
        name=os.path.basename(video).lower()
        if any(k in name for k in ("yawn", "normal", "talk")):
            extract(video, os.path.join(FRAMES, "yawdd"), 0)
