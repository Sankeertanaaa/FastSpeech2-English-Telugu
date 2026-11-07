from pathlib import Path
import numpy as np

root = Path("preprocessed_data/Telugu")

max_mel = 0
max_tokens = 0
max_duration = 0
longest = None

for split in ["train", "val"]:
    for line in (root / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        uid, speaker, text, raw = line.split("|", 3)

        mel = np.load(root / "mel" / f"{speaker}-mel-{uid}.npy")
        dur = np.load(root / "duration" / f"{speaker}-duration-{uid}.npy")

        if len(mel) > max_mel:
            max_mel = len(mel)
            longest = (split, uid)

        max_tokens = max(max_tokens, len(text))
        max_duration = max(max_duration, int(dur.sum()))

print("Maximum mel frames:", max_mel)
print("Maximum text characters:", max_tokens)
print("Maximum duration sum:", max_duration)
print("Longest mel sample:", longest)