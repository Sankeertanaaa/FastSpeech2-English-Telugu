import re
import unicodedata
from pathlib import Path

import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "Telugu" / "train.csv"
OUT = ROOT / "data" / "Telugu" / "mfa_test"
AUDIO_OUT = OUT / "corpus" / "speaker1"

AUDIO_OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV).head(20)

def clean_text(text):
    text = str(text)
    text = text.replace("వెళ్ళగొట్టడo", "వెళ్ళగొట్టడం")
    text = text.replace("ఎoత", "ఎంత")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[,.?\"“”‘’!;:]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(" ".join(word) for word in text.split())

for row in df.itertuples():
    audio, sr = sf.read(row.audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        audio = resample_poly(audio, 320, 441)

    audio = np.clip(audio, -1.0, 1.0)

    wav_path = AUDIO_OUT / f"{row.utterance_id}.wav"
    lab_path = AUDIO_OUT / f"{row.utterance_id}.lab"

    sf.write(wav_path, audio, 16000, subtype="PCM_16")
    lab_path.write_text(clean_text(row.text), encoding="utf-8")

print("Prepared utterances:", len(df))
print("Corpus:", OUT / "corpus")
print("Audio sample rate: 16000 Hz")
print("Original audio files were not modified.")
print("\nFirst 3 alignment transcripts:")
for p in sorted(AUDIO_OUT.glob("*.lab"))[:3]:
    print(p.name, ":", p.read_text(encoding="utf-8"))