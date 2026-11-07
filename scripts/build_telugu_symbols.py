
from pathlib import Path
import pandas as pd
import json

BASE = Path(__file__).resolve().parent.parent

train_path = BASE / "data" / "Telugu" / "train.csv"
output_path = BASE / "data" / "Telugu" / "phone_to_id.json"

df = pd.read_csv(train_path)

characters = set()

for text in df["text"].dropna().astype(str):
    for char in text:
        if not char.isspace():
            characters.add(char)

symbols = ["<pad>", "<unk>", "<space>"] + sorted(characters)

char_to_id = {
    symbol: index
    for index, symbol in enumerate(symbols)
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(char_to_id, f, ensure_ascii=False, indent=2)

print("Telugu character mapping created.")
print("Training utterances:", len(df))
print("Total symbols:", len(symbols))
print("Padding ID:", char_to_id["<pad>"])
print("Unknown ID:", char_to_id["<unk>"])
print("Saved:", output_path)
