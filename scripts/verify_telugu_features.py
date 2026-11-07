from pathlib import Path
import numpy as np
import json

p = Path("preprocessed_data/Telugu")
v = json.load(open("data/Telugu/phone_to_id.json", encoding="utf-8"))

errors = []
checked = 0

for split in ["train", "val"]:
    lines = (p / f"{split}.txt").read_text(encoding="utf-8").splitlines()

    for line in lines:
        uid, speaker, text, raw = line.split("|", 3)

        try:
            mel = np.load(p / "mel" / f"{speaker}-mel-{uid}.npy")
            pitch = np.load(p / "pitch" / f"{speaker}-pitch-{uid}.npy")
            energy = np.load(p / "energy" / f"{speaker}-energy-{uid}.npy")
            dur = np.load(p / "duration" / f"{speaker}-duration-{uid}.npy")

            tokens = [
                v.get("<space>" if c.isspace() else c, v["<unk>"])
                for c in text
            ]

            if mel.ndim != 2 or mel.shape[1] != 80:
                errors.append((uid, "mel shape", mel.shape))
            if len(pitch) != len(mel):
                errors.append((uid, "pitch length"))
            if len(energy) != len(mel):
                errors.append((uid, "energy length"))
            if len(dur) != len(tokens):
                errors.append((uid, "duration/token length"))
            if int(dur.sum()) != len(mel):
                errors.append((uid, "duration sum"))
            if not all(np.isfinite(x).all() for x in [mel, pitch, energy, dur]):
                errors.append((uid, "non-finite values"))

            checked += 1

        except Exception as e:
            errors.append((uid, str(e)))

print("Samples checked:", checked)
print("Errors:", len(errors))
for error in errors[:15]:
    print(error)

print("RESULT:", "PASS" if not errors and checked == 1817 else "REVIEW")