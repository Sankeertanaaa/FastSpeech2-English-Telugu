
import argparse
import hashlib
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from tqdm import tqdm


SOURCES = {
    "English": {
        "repo": "ai4bharat/Svarah",
        "split": "test",
        "audio_key": "audio_filepath",
    },
    "Telugu": {
        "repo": "SPRINGLab/IndicTTS_Telugu",
        "split": "train",
        "audio_key": "audio",
    },
}


def decode_audio(audio):
    if not isinstance(audio, dict):
        return None, None

    raw_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    try:
        if raw_bytes:
            samples, sr = sf.read(
                io.BytesIO(raw_bytes), dtype="float32"
            )
        elif audio_path and Path(audio_path).exists():
            samples, sr = sf.read(audio_path, dtype="float32")
        else:
            return None, None
    except Exception as exc:
        print("Audio decoding failed:", exc)
        return None, None

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    if samples.ndim != 1 or len(samples) == 0:
        return None, None

    return samples.astype(np.float32), int(sr)


def resample_audio(samples, old_sr, new_sr):
    if old_sr == new_sr:
        return samples

    divisor = math.gcd(old_sr, new_sr)
    return resample_poly(
        samples,
        new_sr // divisor,
        old_sr // divisor
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", required=True,
        choices=["English", "Telugu"]
    )
    parser.add_argument("--target-root", default="data")
    parser.add_argument("--train-hours", type=float, default=5)
    parser.add_argument("--val-hours", type=float, default=1)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    language = args.language
    source = SOURCES[language]
    target_sr = args.sample_rate

    output_dir = Path(args.target_root) / language
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {language} dataset...")
    print("Repository:", source["repo"])
    print("Split:", source["split"])

    ds = load_dataset(
        source["repo"],
        split=source["split"],
        streaming=True
    )

    audio_key = source["audio_key"]
    ds = ds.cast_column(audio_key, Audio(decode=False))
   
    # Stream sequentially to avoid large PyArrow shuffle buffers on Windows.
    # The selected subset will be documented in the technical report.

    targets = {
        "val": args.val_hours * 3600,
        "train": args.train_hours * 3600,
    }
    collected = {"val": [], "train": []}
    seconds = {"val": 0.0, "train": 0.0}
    seen = set()
    skipped = 0

    progress = tqdm(desc=f"Downloading {language}")

    for index, row in enumerate(ds):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue

        samples, sr = decode_audio(row.get(audio_key))
        if samples is None or sr <= 0:
            skipped += 1
            continue

        duration = len(samples) / sr
        if duration < 0.35 or duration > 30:
            skipped += 1
            continue

        samples = resample_audio(samples, sr, target_sr)

        audio_hash = hashlib.sha256(
            samples.tobytes()
        ).hexdigest()

        if audio_hash in seen:
            skipped += 1
            continue
        seen.add(audio_hash)

        split = None
        for candidate in ("val", "train"):
            if seconds[candidate] < targets[candidate]:
                split = candidate
                break

        if split is None:
            break

        uid = f"{language.lower()}_{audio_hash[:16]}"
        wav_path = audio_dir / f"{uid}.wav"

        sf.write(
            str(wav_path),
            samples,
            target_sr,
            subtype="PCM_16"
        )

        actual_duration = len(samples) / target_sr

        collected[split].append({
            "utterance_id": uid,
            "text": text.strip(),
            "audio_path": str(wav_path.resolve()),
            "duration_sec": actual_duration,
            "sampling_rate": target_sr,
        })

        seconds[split] += actual_duration
        progress.update(1)
        progress.set_postfix(
            split=split,
            val_min=round(seconds["val"] / 60, 1),
            train_min=round(seconds["train"] / 60, 1)
        )

    progress.close()

    for split, rows in collected.items():
        csv_path = output_dir / f"{split}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        print(
            f"{split}: {len(rows)} utterances, "
            f"{seconds[split] / 3600:.3f} hours"
        )
        print("Saved:", csv_path)

    print("Skipped:", skipped)
    print("Total unique audio files:", len(seen))

    if seconds["val"] < targets["val"] * 0.95:
        print("WARNING: Validation target was not reached.")

    if seconds["train"] < targets["train"] * 0.95:
        print("WARNING: Training target was not reached.")

    print("Dataset download complete.")


if __name__ == "__main__":
    main()
