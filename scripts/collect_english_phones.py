
from pathlib import Path
from collections import Counter
import pandas as pd
import tgt

BASE = Path(".")
SPLITS = [
    ("train", BASE / "data/English/train_aligned_filtered.csv",
     BASE / "alignment/English/aligned_train"),
    ("val", BASE / "data/English/val_aligned_filtered.csv",
     BASE / "alignment/English/aligned_val"),
]

phone_counts = Counter()
missing = 0
total_files = 0

for split, csv_path, alignment_dir in SPLITS:
    df = pd.read_csv(csv_path)
    print(f"\nProcessing {split}: {len(df)} utterances")

    for uid in df["utterance_id"].astype(str):
        matches = list(alignment_dir.rglob(f"{uid}.TextGrid"))

        if not matches:
            missing += 1
            continue

        grid = tgt.io.read_textgrid(str(matches[0]))
        tier = grid.get_tier_by_name("phones")

        for interval in tier.intervals:
            label = interval.text.strip()

            if label:
                phone_counts[label] += 1

        total_files += 1

print("\n========== PHONE INVENTORY ==========")
print("TextGrids inspected:", total_files)
print("Missing TextGrids:", missing)
print("Unique phone labels:", len(phone_counts))

for phone, count in sorted(phone_counts.items()):
    print(f"{repr(phone):15} {count}")

output = BASE / "data/English/phone_inventory.csv"

pd.DataFrame(
    sorted(phone_counts.items()),
    columns=["phone", "count"]
).to_csv(output, index=False, encoding="utf-8")

print("\nSaved:", output)
