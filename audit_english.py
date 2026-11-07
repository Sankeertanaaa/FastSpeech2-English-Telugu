from pathlib import Path

p = Path("preprocessed_data/English")

for name in ["train.txt", "val.txt"]:
    lines = (p / name).read_text(encoding="utf-8").splitlines()

    print(f"\n--- {name} ---")
    print("Total rows:", len(lines))
    print("First row:", repr(lines[0]) if lines else "EMPTY")

    bad = []
    for i, line in enumerate(lines, 1):
        if len(line.split("|", 3)) != 4:
            bad.append((i, repr(line[:200])))

    print("Malformed rows:", len(bad))
    for item in bad[:10]:
        print(item)