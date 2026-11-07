import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Telugu"
OUT = DATA / "mfa_val_corpus" / "speaker1"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "val.csv")

def clean_text(text):
    text = str(text)
    text = text.replace("వెళ్ళగొట్టడo", "వెళ్ళగొట్టడం")
    text = text.replace("ఎoత", "ఎంత")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[,.?\"“”‘’!;:]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(" ".join(word) for word in text.split())

errors = []

for row in df.itertuples():
    try:
        audio, sr = sf.read(row.audio_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            audio = resample_poly(audio, 320, 441)

        audio = np.clip(audio, -1.0, 1.0)

        uid = str(row.utterance_id)
        sf.write(OUT / f"{uid}.wav", audio, 16000, subtype="PCM_16")
        (OUT / f"{uid}.lab").write_text(
            clean_text(row.text), encoding="utf-8"
        )

    except Exception as e:
        errors.append((row.utterance_id, str(e)))

print("Validation utterances:", len(df))
print("Prepared:", len(list(OUT.glob("*.wav"))))
print("Errors:", len(errors))
if errors:
    print(errors[:10])