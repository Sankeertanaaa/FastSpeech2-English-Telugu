import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="NPZ exported by the inference hook")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-id", default="sample")
    args = ap.parse_args()
    data = np.load(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        ("duration", "duration", "Predicted duration (mel frames)"),
        ("pitch", "pitch", "Predicted F0 / pitch"),
        ("energy", "energy", "Predicted energy"),
    ]
    for key, filename, ylabel in fields:
        if key not in data:
            raise KeyError(f"{key} missing in {args.input}; export it from model output first")
        values = np.asarray(data[key]).squeeze()
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(values.size), values)
        plt.xlabel("Phoneme / frame index")
        plt.ylabel(ylabel)
        plt.title(f"{args.sample_id}: {key}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out / f"{filename}.png", dpi=180)
        plt.close()
        print(f"Saved {out / f'{filename}.png'}")

if __name__ == "__main__":
    main()
