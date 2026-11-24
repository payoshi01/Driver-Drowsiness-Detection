# Lightweight Four-State Driver Monitor

Real-time driver monitoring from a single RGB webcam. Each frame is classified into one of four states:
**Focused, Drowsy, Yawning, Looking Away**. The pipeline uses MediaPipe Face Mesh, a small PyTorch crop CNN
(about 138k parameters), post-training INT8 quantization, and a temporal decision layer (majority vote + PERCLOS-style alarm).

Class order is fixed everywhere: `Focused=0, Drowsy=1, Yawning=2, LookingAway=3`.

## Main results (held-out, subject-disjoint test split, 21,874 frames)

| Model | Params | Accuracy | Macro-F1 |
|---|---|---|---|
| Crop CNN (selected) | 138k | **0.912** | **0.637** |
| Geometry-only network | 3.7k | 0.849 | 0.633 |
| Rule-based baseline | - | 0.685 | 0.535 |
| Focal-loss CNN | 138k | 0.891 | 0.576 |
| Fusion (crops + geometry) | 142k | 0.893 | 0.595 |
| MobileNetV3-Small | 1.94M | 0.889 | 0.561 |

- **Drowsy is the weakest class** (best F1 0.41) and is mostly confused with Looking Away. Expect false Drowsy readings in the demo.
- **INT8 quantization** shrinks the crop CNN from 0.572 MB to 0.195 MB (2.9x) with unchanged macro-F1 (0.637 to 0.638), but gives no speed-up on an x86 laptop CPU (Face Mesh dominates; about 88-95 FPS end-to-end). MobileNetV3 degrades under INT8 (0.561 to 0.480).
- **Latency / size benchmark** (batch size 1, one thread, medians of three runs): CNN FP32 1.33 ms, INT8 2.10 ms; MobileNetV3 FP32 7.99 MB, INT8 2.54 MB.
- **Synthetic low light** (L1-L3): Face Mesh finds a face in 99.1% / 93.2% / 5.3% of frames.
- Rule-based thresholds (validation-calibrated, `thresholds.json`): EAR 0.26, MAR 0.35, yaw 45 deg, pitch 35 deg.

See `results/` for the confusion matrices and CSV files behind these numbers.

## Repository layout

```
config.py            constants (paths, classes, thresholds, temporal-layer settings)
features.py          Face Mesh features (9 geometric values) + 64x64 eye/mouth crops
build_features.py    features for train/val/test and for the low-light sets
rule_based.py        rule-based baseline (EAR/MAR/yaw/pitch thresholds), writes thresholds.json
models.py            geometry network, crop CNN, fusion, MobileNetV3 variants
train.py             training and ablations
quantize.py          INT8 post-training quantization (eager for CNN, FX for MobileNetV3)
benchmark.py         size / latency / throughput benchmark, FP32 vs INT8
utils.py             data loading, metrics, batching
app.py               desktop demo (OpenCV window) with the temporal Smoother
web_app.py           browser demo (Flask), uses Smoother from app.py
templates/index.html browser demo page
run_all.sh           whole pipeline after raw data is in place
data_tools/          record.py, extract_frames.py, build_labels.py, make_lowlight.py, make_alarm.py
tests/               subject-leakage and class-coverage tests
dataset/  data/  checkpoints/  results/ 
```

Large data, features and model files (`dataset/`, `data/`, `checkpoints/*.pt`) are not tracked in git.
Keep the data and checkpoints on a shared drive and place them in these folders locally.

## 0. Setup

Python 3.11 is recommended (the project was run on 3.11 with MediaPipe 0.10.14 and PyTorch 2.4.1).

```bash
python -m venv venv
venv\Scripts\activate                  # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
pip install flask                      # needed for the browser demo (also add it to requirements.txt)
python data_tools/make_alarm.py        # creates alarm.wav for the desktop beep
```

## 1. Data

1. Download **MRL Eye** and unzip it into `dataset/raw/mrl/`.
2. Download **YawDD** (Mirror set) into `dataset/raw/yawdd/`. Only files whose names contain Normal/Talking/Yawning are used; edit the regex in `data_tools/build_labels.py` if your names differ.
3. Own recordings go in `dataset/raw/own/`. Keys 1-4 start a 30 s clip:
   ```bash
   python data_tools/record.py p01 bright noglasses
   python data_tools/record.py p01 dim    noglasses
   ```
   Our own set has seven subjects with clips for Drowsy and Looking Away, three of them recorded in a dark room lit only by a study lamp. Drowsy and Looking Away clips also include YouTube subjects.
4. Frames, labels, subject-level split and the low-light copies of the test split:
   ```bash
   python data_tools/extract_frames.py
   python data_tools/build_labels.py --freeze     # writes dataset/labels.csv, checks leakage
   python -m pytest -q                            # leakage + class-coverage tests
   python data_tools/make_lowlight.py             # L1/L2/L3
   ```

The split is subject-disjoint: no person appears in more than one of train/val/test.
Frames in yawning clips whose MAR is below `YAWN_MAR_MIN` (0.35) are relabelled Focused (in the feature file only; `labels.csv` stays frozen).

## 2. Features and rule-based baseline

```bash
python build_features.py
python build_features.py --labels dataset/labels_lowlight.csv --out data/features_lowlight.npz --ref data/features.npz
python rule_based.py        # calibrates on val, reports on test and low-light, writes thresholds.json
```

## 3. Training and ablations

```bash
python train.py --pretrain-eye                  # optional MRL initialisation of the eye encoder
python train.py --models geo cnn fusion         # geometry-only / crop-only / fused
python train.py --models mnv3 fusion_mnv3 --pretrained
```

Outputs: `checkpoints/fp32/*.pt`, `results/ablation.csv`, `results/cm_*.png`, `results/ablation_*.json`. Seed is 42.

## 4. Quantization and benchmark

```bash
python quantize.py --model cnn  --mode eager        # crop CNN -> INT8
python quantize.py --model mnv3 --mode fx           # MobileNetV3 via FX graph mode
python benchmark.py --name cnn --precision fp32       --ckpt checkpoints/fp32/cnn.pt
python benchmark.py --name cnn --precision int8_eager --ckpt checkpoints/int8/cnn_eager.pt
python benchmark.py --plot                           # results/benchmark_summary.csv + charts
```

`--ckpt` is required. Run each benchmark in a fresh process with other apps closed, and repeat three times.
On ARM devices (for example Raspberry Pi) the engine switches to `qnnpack` automatically (`config.QENGINE`).

## 5. Demos

Both demos use the same temporal decision layer (`Smoother` in `app.py`) and need a webcam. Run one at a time.

### Desktop (OpenCV window)
```bash
python app.py --ckpt checkpoints\int8\cnn_eager.pt
```
Press `q` to quit. Options: `--camera N`, `--threads N`, `--no-alarm`. The overlay shows the smoothed state, PERCLOS, FPS, EAR, MAR and yaw; a red frame and sound appear when the alarm fires.

### Browser (Flask)
```bash
python web_app.py --ckpt checkpoints\int8\cnn_eager.pt
```
Open `http://127.0.0.1:5000` and press **Start Camera**. The page sends webcam frames to a local server that runs the same pipeline. It shows the status, live EAR/MAR/yaw/pitch, PERCLOS, and blink and yawn counts. The server only listens on `127.0.0.1`. Use `--port N` if 5000 is busy.
A blink is counted when EAR stays below the threshold in `thresholds.json` for 0.05-1.0 s; a yawn is counted on each change of state to Yawning.

### Temporal decision layer

| Rule | Value |
|---|---|
| State | majority vote over the last 9 frames |
| Alarm | 45 consecutive Drowsy frames, or PERCLOS > 0.20 over a 30 s window with more than 10 s of data |
| Release | 15 non-Drowsy frames in a row **and** PERCLOS < 0.16 |
| No face | after 15 frames the state becomes Looking Away |

The rules are counted in frames, so delays in seconds depend on the frame rate (1.5 s is 45 frames only at 30 fps).
Because PERCLOS covers 30 s, the alarm can stay on for roughly 16-25 s after the eyes reopen.

## Known limitations

- Drowsy recall is low; false Drowsy readings are common in the live demo, especially with poor light, glasses or a downward head tilt.
- The own set has few subjects, so results for Drowsy and Looking Away are indicative only. 620 Drowsy/Looking Away labels were refined manually by a single reviewer, with no measurable gain.
- INT8 gives no speed-up on x86; it is expected to matter more on ARM.
- Neither demo has been evaluated for alarm delay or false alarms per minute, and the browser version's throughput was not benchmarked.
- Real low-light performance is not evaluated (only synthetic L1-L3).

## Rules for contributors

- Never edit `dataset/raw/`; never tune on the test split; freeze `labels.csv` before training.
- Crops, colour order and feature standardisation live in `features.py` and inside the model, so training and the demos cannot drift apart.
- Frames where no face is found are excluded from training and evaluation.
- `config.py` must define both `PERCLOSE_THRESH` and the alias `PERCLOS_THRESH` (used by `Smoother`) and `FEAT["pitch"]` (used by `web_app.py`).
- Do not push `dataset/`, `data/` or `*.pt` files; check `git count-objects -vH` before pushing.

## Datasets

- MRL Eye dataset (eye open/closed, infrared)
- YawDD (Yawning Detection Dataset, in-vehicle)
- Own webcam recordings and YouTube videos

Cite the original dataset papers when you use them.

## Team Members

- Payoshi Gupta
- Aayushi Vinod
- Anshika Singh
- Tanisha
- Amit Yadav