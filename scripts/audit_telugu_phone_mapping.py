from pathlib import Path
import pandas as pd
import tgt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/Telugu"

def audit(csv_path, corpus_path, aligned_path):
    df = pd.read_csv(csv_path)
    checked = 0
    missing = 0
    mismatches = []

    for row in df.itertuples():
        uid = str(row.utterance_id)
        lab = corpus_path / (uid + ".lab")
        grids = list(aligned_path.rglob(uid + ".TextGrid"))

        if not grids or not lab.exists():
            missing += 1
            continue

        expected = lab.read_text(encoding="utf-8").split()
        grid = tgt.io.read_textgrid(str(grids[0]))
        tier = grid.get_tier_by_name("phones")

        actual = [
            x.text if x.text != "sil" else "<space>"
            for x in tier.intervals
            if x.text not in ("",)
        ]

        while actual and actual[0] == "<space>":
            actual.pop(0)

        while actual and actual[-1] == "<space>":
            actual.pop()

        if expected != actual:
            mismatches.append(uid)

        checked += 1

    print("CSV:", csv_path.name)
    print("Checked:", checked)
    print("Missing:", missing)
    print("Sequence mismatches:", len(mismatches))
    print("Examples:", mismatches[:10])


audit(
    DATA / "train.csv",
    DATA / "mfa_train_corpus/speaker1",
    DATA / "mfa_train_aligned_fixed"
)

print("\n--- VALIDATION ---")

audit(
    DATA / "val.csv",
    DATA / "mfa_val_corpus/speaker1",
    DATA / "mfa_val_aligned"
)