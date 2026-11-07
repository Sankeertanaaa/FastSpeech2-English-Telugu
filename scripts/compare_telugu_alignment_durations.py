from pathlib import Path
import pandas as pd
import tgt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/Telugu"

df = pd.read_csv(DATA / "train_aligned_candidates.csv")
quality = pd.read_csv(DATA / "train_alignment_quality.csv")

df = df.merge(quality, on="utterance_id")

df["duration_difference"] = (
    df["duration"] - df["duration_sec"]
)

df["duration_ratio"] = (
    df["duration"] / df["duration_sec"]
)

print("Original audio duration (mean):",
      round(df.duration_sec.mean(), 2))
print("Aligned duration (mean):",
      round(df.duration.mean(), 2))
print("Mean duration ratio:",
      round(df.duration_ratio.mean(), 3))
print("Median duration ratio:",
      round(df.duration_ratio.median(), 3))
print("Ratio below 0.90:",
      (df.duration_ratio < 0.90).sum())
print("Ratio above 1.10:",
      (df.duration_ratio > 1.10).sum())

print("\nLowest duration ratios:")
print(df.nsmallest(10, "duration_ratio")[
    ["utterance_id", "duration_sec", "duration",
     "duration_ratio", "silence_ratio"]
].to_string(index=False))

print("\nHighest duration ratios:")
print(df.nlargest(10, "duration_ratio")[
    ["utterance_id", "duration_sec", "duration",
     "duration_ratio", "silence_ratio"]
].to_string(index=False))