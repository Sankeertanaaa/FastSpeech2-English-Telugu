from pathlib import Path
import tgt

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "data/Telugu/mfa_test/corpus/speaker1"
GRID_DIR = ROOT / "data/Telugu/mfa_test/aligned"

checked = 0
mismatches = []

for lab in sorted(LAB_DIR.glob("*.lab")):
    matches = list(GRID_DIR.rglob(lab.stem + ".TextGrid"))

    if not matches:
        mismatches.append((lab.stem, "MISSING TEXTGRID"))
        continue

    expected = lab.read_text(encoding="utf-8").split()
    grid = tgt.io.read_textgrid(str(matches[0]))
    tier = grid.get_tier_by_name("words")

    actual = [
        x.text for x in tier.intervals
        if x.text not in ("", "<eps>")
    ]

    checked += 1

    if expected != actual:
        mismatches.append((
            lab.stem,
            f"Expected {len(expected)} graphemes, got {len(actual)}",
            expected[:20],
            actual[:20]
        ))

print("Files checked:", checked)
print("Sequence mismatches:", len(mismatches))

for item in mismatches[:10]:
    print(item)

if checked and not mismatches:
    print("PASS: All aligned grapheme sequences exactly match transcripts.")