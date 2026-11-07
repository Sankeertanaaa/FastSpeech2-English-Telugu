
from pathlib import Path
import pandas as pd
import json

BASE = Path(".")
inventory_path = BASE / "data/English/phone_inventory.csv"
output_path = BASE / "data/English/phone_to_id.json"

df = pd.read_csv(inventory_path)

phones = sorted(
    phone for phone in df["phone"].astype(str).tolist()
    if phone not in ("spn", "<pad>", "<sil>", "<unk>")
)

symbols = ["<pad>", "<sil>", "<unk>"] + phones

phone_to_id = {
    phone: index
    for index, phone in enumerate(symbols)
}

with output_path.open("w", encoding="utf-8") as file:
    json.dump(phone_to_id, file, ensure_ascii=False, indent=2)

print("Phone mapping created.")
print("Total symbols:", len(symbols))
print("Padding ID:", phone_to_id["<pad>"])
print("Silence ID:", phone_to_id["<sil>"])
print("Unknown/SPN ID:", phone_to_id["<unk>"])
print("Saved to:", output_path)
