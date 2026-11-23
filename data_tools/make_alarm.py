import os, sys, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import ALARM_WAV
sr = 22050; t = np.linspace(0, 0.25, int(sr * 0.25), False)
beep = (np.sin(2 * np.pi * 880 * t) * 0.6 * 32767).astype(np.int16)
sig = np.concatenate([beep, np.zeros_like(beep)] * 3)
with wave.open(ALARM_WAV, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(sig.tobytes())
print("wrote", ALARM_WAV)