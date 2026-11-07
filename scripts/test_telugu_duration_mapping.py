
import re
import unicodedata
import yaml
import numpy as np
import pandas as pd
import librosa
import tgt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio as Audio

DATA = ROOT / "data" / "Telugu"

uid = "telugu_2fa501aab08cff56"
SR = 22050
HOP = 256


def clean_words(text):
    text = str(text)
    text = text.replace("వెళ్ళగొట్టడo", "వెళ్ళగొట్టడం")
    text = text.replace("ఎoత", "ఎంత")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[,.?\"“”‘’!;:]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


df = pd.read_csv(DATA / "train.csv")
row = df.loc[df["utterance_id"] == uid].iloc[0]

words = clean_words(row["text"])

tokens = []
boundary_positions = set()
grapheme_count = 0

for i, word in enumerate(words):
    if i > 0:
        boundary_positions.add(grapheme_count)
        tokens.append("<space>")

    tokens.extend(list(word))
    grapheme_count += len(word)

graphemes = [token for token in tokens if token != "<space>"]

grid_path = next(
    (DATA / "mfa_train_aligned_fixed").rglob(uid + ".TextGrid")
)

grid = tgt.io.read_textgrid(str(grid_path))
tier = grid.get_tier_by_name("phones")

durations = [0] * len(tokens)
seen = 0
start_time = None
end_time = None
mismatch = []

space_indices = [
    i for i, token in enumerate(tokens) if token == "<space>"
]

grapheme_indices = [
    i for i, token in enumerate(tokens) if token != "<space>"
]

for interval in tier.intervals:
    label = interval.text
    s = interval.start_time
    e = interval.end_time

    if label in ("", "sil", "sp", "spn"):
        if start_time is None or seen >= len(graphemes):
            continue

        frames = (
            int(np.round(e * SR / HOP))
            - int(np.round(s * SR / HOP))
        )

        if seen in boundary_positions:
            boundary_idx = next(
                (
                    idx for idx in space_indices
                    if sum(
                        1 for x in tokens[:idx]
                        if x != "<space>"
                    ) == seen
                ),
                None
            )

            if boundary_idx is not None:
                durations[boundary_idx] += frames

        elif seen > 0:
            durations[grapheme_indices[seen - 1]] += frames

        continue

    if seen >= len(graphemes):
        mismatch.append(("unexpected", label))
        continue

    expected = graphemes[seen]

    if label != expected:
        mismatch.append((expected, label))

    token_idx = grapheme_indices[seen]

    frames = (
        int(np.round(e * SR / HOP))
        - int(np.round(s * SR / HOP))
    )

    durations[token_idx] += frames

    if start_time is None:
        start_time = s

    end_time = e
    seen += 1


print("Transcript graphemes:", len(graphemes))
print("Aligned graphemes:", seen)
print("Sequence mismatches:", len(mismatch))
print("Duration entries:", len(durations))
print("Zero-duration entries:", sum(x == 0 for x in durations))
print("Duration sum:", sum(durations))

if start_time is None or end_time is None:
    raise ValueError("No valid grapheme intervals found.")

wav, _ = librosa.load(row["audio_path"], sr=SR)
wav = wav.astype(np.float32)

with open(
    ROOT / "configs" / "Telugu" / "preprocess.yaml",
    encoding="utf-8"
) as f:
    config = yaml.safe_load(f)

stft = Audio.stft.TacotronSTFT(
    config["preprocessing"]["stft"]["filter_length"],
    HOP,
    config["preprocessing"]["stft"]["win_length"],
    config["preprocessing"]["mel"]["n_mel_channels"],
    SR,
    config["preprocessing"]["mel"]["mel_fmin"],
    config["preprocessing"]["mel"]["mel_fmax"],
)

full_mel, full_energy = Audio.tools.get_mel_from_wav(
    wav, stft
)

start_frame = int(np.round(start_time * SR / HOP))
end_frame = int(np.round(end_time * SR / HOP))

mel = full_mel[:, start_frame:end_frame]
energy = full_energy[start_frame:end_frame]

print("\nZERO-DURATION TOKENS:")
for token, duration in zip(tokens, durations):
    if duration == 0:
        print(repr(token))

print("\nMEL AND DURATION CHECK")
print("Full mel frames:", full_mel.shape[1])
print("Crop frame boundaries:", start_frame, end_frame)
print("Mel shape:", mel.shape)
print("Mel frames:", mel.shape[1])
print("Duration sum:", sum(durations))
print("Duration/mel difference:", mel.shape[1] - sum(durations))
print("Mapping mismatches:", mismatch[:10])

if mismatch:
    print("WARNING: Transcript and alignment sequences do not match.")

if len(durations) != len(tokens):
    raise ValueError("Duration and token lengths differ.")
