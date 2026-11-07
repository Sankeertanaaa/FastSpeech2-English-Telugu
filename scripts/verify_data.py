import argparse
from pathlib import Path
import pandas as pd
import soundfile as sf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    args = p.parse_args()
    for lang in ("English", "Telugu"):
        folder = Path(args.root) / lang
        print(f"\n[{lang}]")
        frames = {}
        ids = {}
        for split in ("train", "val"):
            path = folder / f"{split}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            required = {"utterance_id", "text", "audio_path", "duration_sec", "sampling_rate"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            if df.empty:
                raise ValueError(f"{path}: no rows")
            total = 0.0
            for audio_path in df["audio_path"]:
                info = sf.info(audio_path)
                if info.channels != 1:
                    raise ValueError(f"Expected mono audio: {audio_path}")
                if info.samplerate != 22050:
                    raise ValueError(f"Expected 22050 Hz: {audio_path} ({info.samplerate})")
                total += info.duration
            frames[split] = df
            ids[split] = set(df["utterance_id"].astype(str))
            print(f"{split}: {len(df)} clips, {total/3600:.3f} hours")
        overlap = ids["train"] & ids["val"]
        if overlap:
            raise RuntimeError(f"Train/validation overlap in {lang}: {len(overlap)} IDs")
        print("Utterance IDs: no overlap")
        print("Text/audio rows:", len(frames["train"]) + len(frames["val"]))

if __name__ == "__main__":
    main()
