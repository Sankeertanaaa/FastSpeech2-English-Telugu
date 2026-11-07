from pathlib import Path
import pandas as pd
import numpy as np
import tgt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/Telugu"
CSV = pd.read_csv(DATA / "val.csv")
CORPUS = DATA / "mfa_val_corpus/speaker1"
ALIGNED = DATA / "mfa_val_aligned"

missing = []
mismatches = []
ratios = []
silence_ratios = []

for row in CSV.itertuples():
    uid = str(row.utterance_id)
    lab = CORPUS / (uid + ".lab")
    matches = list(ALIGNED.rglob(uid + ".TextGrid"))

    if not matches:
        missing.append(uid)
        continue

    expected = lab.read_text(encoding="utf-8").split()
    grid = tgt.io.read_textgrid(str(matches[0]))
    words = grid.get_tier_by_name("words")
    phones = grid.get_tier_by_name("phones")

    actual = [
        x.text for x in words.intervals
        if x.text not in ("", "<eps>")
    ]

    if expected != actual:
        mismatches.append(uid)

    aligned_duration = phones.end_time
    ratios.append(aligned_duration / row.duration_sec)

    silence = sum(
        x.end_time - x.start_time
        for x in phones.intervals if x.text == "sil"
    )
    silence_ratios.append(
        silence / aligned_duration if aligned_duration else 1
    )

print("Input utterances:", len(CSV))
print("TextGrids:", len(ratios))
print("Missing:", len(missing))
print("Sequence mismatches:", len(mismatches))
print("Mean duration ratio:", round(np.mean(ratios), 4))
print("Median duration ratio:", round(np.median(ratios), 4))
print("Ratios outside 0.99-1.01:",
      sum(r < 0.99 or r > 1.01 for r in ratios))
print("Mean silence ratio:", round(np.mean(silence_ratios), 3))
print("Median silence ratio:", round(np.median(silence_ratios), 3))

pd.DataFrame({"utterance_id": missing}).to_csv(
    DATA / "mfa_val_missing.csv", index=False
)
pd.DataFrame({"utterance_id": mismatches}).to_csv(
    DATA / "mfa_val_mismatches.csv", index=False
)

print("Saved missing and mismatch reports.")