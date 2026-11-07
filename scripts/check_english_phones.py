import json
import tgt
import pandas as pd
from pathlib import Path

vocab_path = Path("data/English/phone_to_id.json")

with vocab_path.open("r", encoding="utf-8") as f:
    vocab = json.load(f)

phones = set()

for split in ["train", "val"]:
    csv_path = Path(f"data/English/{split}_aligned_filtered.csv")
    df = pd.read_csv(csv_path)

    folder_name = "aligned_train" if split == "train" else "aligned_val"
    folder = Path("alignment/English") / folder_name / "speaker_1"

    for uid in df["utterance_id"]:
        grid_path = folder / f"{uid}.TextGrid"
        tg = tgt.io.read_textgrid(str(grid_path))

        tier = tg.get_tier_by_name("phones")

        for interval in tier.intervals:
            phone = interval.text.strip()

            if phone and phone not in ["sil", "sp", "spn"]:
                phones.add(phone)

missing = phones - set(vocab)

print("Unique alignment phones:", len(phones))
print("Vocabulary size:", len(vocab))
print("Missing phones:", sorted(missing))