from pathlib import Path
import re
import csv

for split in ["train", "val"]:
    folder = Path(f"alignment/English/aligned_{split}")
    results = []

    for path in folder.rglob("*.TextGrid"):
        content = path.read_text(encoding="utf-8")
        tiers = re.split(r"item\s*\[\d+\]\s*:", content)

        total_duration = 0.0
        spn_duration = 0.0
        spn_count = 0

        for tier in tiers:
            if not re.search(r'name\s*=\s*"phones"', tier):
                continue

            intervals = re.split(r"intervals\s*\[\d+\]\s*:", tier)

            for interval in intervals[1:]:
                start = re.search(r"xmin\s*=\s*([\d.]+)", interval)
                end = re.search(r"xmax\s*=\s*([\d.]+)", interval)
                label = re.search(r'text\s*=\s*"(.*?)"', interval, re.S)

                if not (start and end and label):
                    continue

                duration = float(end.group(1)) - float(start.group(1))
                phone = label.group(1).strip()

                if duration > 0:
                    total_duration += duration

                    if phone.lower() == "spn":
                        spn_duration += duration
                        spn_count += 1

        if total_duration > 0:
            percentage = 100 * spn_duration / total_duration
            results.append((percentage, path.stem, spn_count, spn_duration, total_duration))

    results.sort(reverse=True)

    print(f"\n===== {split.upper()} =====")
    print("Utterances audited:", len(results))
    print("Utterances with >20% SPN:",
          sum(x[0] > 20 for x in results))
    print("Utterances with >50% SPN:",
          sum(x[0] > 50 for x in results))

    print("\nTop 15 by SPN duration:")
    for pct, name, count, unknown, total in results[:15]:
        print(
            f"{name}: {pct:.1f}% SPN, "
            f"{count} intervals, "
            f"{unknown:.2f}s / {total:.2f}s"
        )