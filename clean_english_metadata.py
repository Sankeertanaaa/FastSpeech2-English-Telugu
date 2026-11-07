from pathlib import Path

path = Path("preprocessed_data/English/train.txt")

lines = path.read_text(encoding="utf-8").splitlines()
cleaned = [line for line in lines if line.strip() and line.strip() != "."]

path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")

print("Original rows:", len(lines))
print("Cleaned rows:", len(cleaned))
print("Removed:", len(lines) - len(cleaned))