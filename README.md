# Real-time Driver Drowsiness + Distraction Detection

Browser webcam -> Flask -> MediaPipe Face Mesh (468 landmarks) -> rule-based temporal logic -> live alerts + alarm sound.

## Run
```bash
cd driver-monitor
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt   # use Python 3.9 - 3.12
python app.py
```
Open http://127.0.0.1:5000, click **Start Camera**, allow permission.

## How it detects
| Signal | Metric | Default rule | Result |
|---|---|---|---|
| Eyes closed | Eye Aspect Ratio (EAR) | EAR < 0.21 for 1.2 s | DROWSY |
| Yawning | Mouth Aspect Ratio (MAR) | MAR > 0.60 for 1 s | DROWSY |
| Looking sideways | Nose position across face width | <0.30 or >0.70 for 2 s | DISTRACTED |
| Head down / phone | Nose-eye-chin ratio | >0.62 for 2 s | DISTRACTED |
| No face | - | for 2 s | DISTRACTED |

Calibrate: watch the live metric values with eyes open/closed and edit the constants at the top of `detector.py`.

## Relevant datasets (for validating thresholds or training a CNN)
- **MRL Eye Dataset** - ~85k open/closed eye images (mrl.cs.vsb.cz/eyedataset)
- **YawDD** - drivers yawning/talking videos (IEEE DataPort)
- **NTHU-DDD** - drowsy driver videos, various conditions
- **State Farm Distracted Driver Detection** - 10 classes (texting, phone, drinking...) on Kaggle
- **DMD (Driver Monitoring Dataset)** - multi-modal drowsiness and distraction labels

Upgrade path: train MobileNetV2 on State Farm for phone/texting detection and add its prediction as an extra rule in `DriverMonitor.process`.

## Files
- `app.py` Flask server (`/`, `/predict`)
- `detector.py` landmarks, metrics, temporal state
- `templates/index.html` camera UI, metrics, alarm
- `requirements.txt`

## Troubleshooting
- No camera prompt: use `127.0.0.1`/`localhost` (browsers require it for webcam).
- Glasses/low light cause false EAR readings; lower `EAR_THRESH` to ~0.18.
- Calibrate to your camera angle; adjust `YAW_*`/`PITCH_DOWN` if you sit off-center.
