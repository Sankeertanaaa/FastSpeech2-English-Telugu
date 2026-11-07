import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

PATTERN = re.compile(
    r"(?:Validation )?Step\s+(\d+)(?:/\d+)?,\s+Total Loss:\s+([0-9.eE+-]+),\s+"
    r"Mel Loss:\s+([0-9.eE+-]+),\s+Mel PostNet Loss:\s+([0-9.eE+-]+),\s+"
    r"Pitch Loss:\s+([0-9.eE+-]+),\s+Energy Loss:\s+([0-9.eE+-]+),\s+"
    r"Duration Loss:\s+([0-9.eE+-]+)"
)

LABELS = ["Total", "Mel", "Mel PostNet", "Pitch", "Energy", "Duration"]

def parse(path):
    steps, rows = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = PATTERN.search(line)
        if m:
            steps.append(int(m.group(1)))
            rows.append([float(x) for x in m.groups()[1:]])
    return steps, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--val-log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for label, source, filename in [
        ("Training", args.train_log, "training_loss.png"),
        ("Validation", args.val_log, "validation_loss.png"),
    ]:
        steps, values = parse(source)
        if not steps:
            raise RuntimeError(f"No parseable loss entries found in {source}")
        plt.figure(figsize=(10, 6))
        for i, name in enumerate(LABELS):
            plt.plot(steps, [r[i] for r in values], label=name)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title(f"{label} losses")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / filename, dpi=180)
        plt.close()
        print(f"Saved {out / filename}")

if __name__ == "__main__":
    main()
