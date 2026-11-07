
import csv
from pathlib import Path

import tgt

BASE = Path(__file__).resolve().parent.parent

MAX_SPN_RATIO = 0.20

SPLITS = {
    "train": {
        "csv": BASE / "data" / "English" / "train.csv",
        "aligned": BASE / "alignment" / "English" / "aligned_train",
    },
    "val": {
        "csv": BASE / "data" / "English" / "val.csv",
        "aligned": BASE / "alignment" / "English" / "aligned_val",
    },
}


def find_textgrid(folder, utterance_id):
    direct_path = folder / "speaker_1" / f"{utterance_id}.TextGrid"

    if direct_path.exists():
        return direct_path

    matches = list(folder.rglob(f"{utterance_id}.TextGrid"))

    if matches:
        return matches[0]

    return None


def calculate_spn_ratio(textgrid_path):
    grid = tgt.io.read_textgrid(str(textgrid_path))

    try:
        tier = grid.get_tier_by_name("phones")
    except ValueError:
        return None

    total_duration = 0.0
    spn_duration = 0.0
    usable_phones = 0

    for interval in tier.intervals:
        label = interval.text.strip()
        duration = max(0.0, interval.end_time - interval.start_time)

        if not label:
            continue

        total_duration += duration

        if label.lower() == "spn":
            spn_duration += duration
        elif label.lower() not in {"sil", "sp", "spn"}:
            usable_phones += 1

    if total_duration <= 0 or usable_phones == 0:
        return None

    return spn_duration / total_duration


def process_split(split_name, config):
    input_csv = config["csv"]
    aligned_dir = config["aligned"]

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    if not aligned_dir.exists():
        raise FileNotFoundError(f"Alignment folder not found: {aligned_dir}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "utterance_id" not in fieldnames:
        raise ValueError(f"Missing utterance_id column: {input_csv}")

    retained = []
    rejected = []

    missing_count = 0
    invalid_count = 0
    high_spn_count = 0

    for row in rows:
        utterance_id = row["utterance_id"].strip()

        textgrid_path = find_textgrid(aligned_dir, utterance_id)

        if textgrid_path is None:
            rejected.append({
                **row,
                "rejection_reason": "missing_textgrid",
                "spn_ratio": "",
            })
            missing_count += 1
            continue

        try:
            spn_ratio = calculate_spn_ratio(textgrid_path)
        except Exception as error:
            print(f"Warning: {utterance_id}: {error}")
            spn_ratio = None

        if spn_ratio is None:
            rejected.append({
                **row,
                "rejection_reason": "invalid_or_empty_phone_alignment",
                "spn_ratio": "",
            })
            invalid_count += 1
            continue

        if spn_ratio > MAX_SPN_RATIO:
            rejected.append({
                **row,
                "rejection_reason": "high_spn_ratio",
                "spn_ratio": f"{spn_ratio:.6f}",
            })
            high_spn_count += 1
            continue

        retained.append(row)

    output_dir = BASE / "data" / "English"
    output_dir.mkdir(parents=True, exist_ok=True)

    retained_path = output_dir / f"{split_name}_aligned_filtered.csv"
    rejected_path = output_dir / f"{split_name}_alignment_rejected.csv"

    with retained_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)

    with rejected_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames + ["rejection_reason", "spn_ratio"],
        )
        writer.writeheader()
        writer.writerows(rejected)

    print()
    print(f"===== ENGLISH {split_name.upper()} =====")
    print(f"Original utterances: {len(rows)}")
    print(f"Retained utterances: {len(retained)}")
    print(f"Rejected utterances: {len(rejected)}")
    print(f"Missing TextGrids: {missing_count}")
    print(f"Invalid phone alignments: {invalid_count}")
    print(f"High SPN ratio (>{MAX_SPN_RATIO:.0%}): {high_spn_count}")
    print(f"Filtered CSV: {retained_path}")
    print(f"Rejected CSV: {rejected_path}")


def main():
    print("Filtering English MFA alignments...")
    print(f"Maximum allowed SPN duration ratio: {MAX_SPN_RATIO:.0%}")

    for split_name, config in SPLITS.items():
        process_split(split_name, config)

    print()
    print("English alignment filtering completed.")


if __name__ == "__main__":
    main()
