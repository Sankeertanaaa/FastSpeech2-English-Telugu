import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import pyworld as pw
import tgt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Telugu"

uid = "telugu_2fa501aab08cff56"
SR = 22050
HOP = 256

df = pd.read_csv(DATA / "train.csv")
row = df.loc[df["utterance_id"] == uid].iloc[0]

grid_path = next(
    (DATA / "mfa_train_aligned_fixed").rglob(uid + ".TextGrid")
)
grid = tgt.io.read_textgrid(str(grid_path))
tier = grid.get_tier_by_name("phones")

start_time = next(
    x.start_time for x in tier.intervals
    if x.text not in ("", "sil", "sp", "spn")
)
end_time = next(
    x.end_time for x in reversed(tier.intervals)
    if x.text not in ("", "sil", "sp", "spn")
)

wav, _ = librosa.load(row["audio_path"], sr=SR)
wav = wav.astype(np.float64)

pitch, times = pw.dio(
    wav,
    SR,
    frame_period=HOP / SR * 1000
)
pitch = pw.stonemask(wav, pitch, times, SR)

start_frame = int(np.round(start_time * SR / HOP))
end_frame = int(np.round(end_time * SR / HOP))
cropped_pitch = pitch[start_frame:end_frame]

print("Full waveform samples:", len(wav))
print("Full pitch frames:", len(pitch))
print("MFA crop frames:", end_frame - start_frame)
print("Cropped pitch frames:", len(cropped_pitch))
print("Nonzero pitch frames:", np.count_nonzero(cropped_pitch))
print("Pitch finite:", np.isfinite(cropped_pitch).all())
print("Pitch min/max:", cropped_pitch.min(), cropped_pitch.max())