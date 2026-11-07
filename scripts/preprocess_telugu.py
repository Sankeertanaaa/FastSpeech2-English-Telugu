import json
import re
import sys
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyworld as pw
import soundfile as sf
import tgt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio as Audio

DATA = ROOT / "data" / "Telugu"
OUT = ROOT / "preprocessed_data" / "Telugu"

SR = 22050
HOP = 256
SPEAKER = "speaker_1"

TRAIN_CSV = DATA / "train.csv"
VAL_CSV = DATA / "val.csv"

TRAIN_GRID = DATA / "mfa_train_aligned_fixed"
VAL_GRID = DATA / "mfa_val_aligned_fixed"

PUNCTUATION = r'[,.?"“”‘’!;:]'


def clean_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Correct Latin 'o' OCR substitutions in Telugu words.
    text = re.sub(
        r"(?<=[\u0C00-\u0C7F])o(?=[\u0C00-\u0C7F\s,?.!;:]|$)",
                        "ం",
        text,
    )

    text = re.sub(PUNCTUATION, "", text)
    return re.sub(r"\s+", " ", text).strip()


def make_tokens(text):
    tokens = []

    for char in text:
        tokens.append("<space>" if char.isspace() else char)

    return tokens


def remove_outliers(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return values

    q25, q75 = np.percentile(values, [25, 75])
    spread = q75 - q25
    lower = q25 - 1.5 * spread
    upper = q75 + 1.5 * spread

    selected = values[(values > lower) & (values < upper)]

    return selected if len(selected) else values


def build_grid_map(folder):
    return {
        path.stem: path
        for path in folder.rglob("*.TextGrid")
    }


def process_one(row, grid_path, stft, vocab):
    uid = str(row.utterance_id)
    raw_text = str(row.text)
    text = clean_text(raw_text)

    if not text:
        raise ValueError("Empty cleaned transcript")

    tokens = make_tokens(text)

    if any(token not in vocab for token in tokens):
        unknown = sorted(set(t for t in tokens if t not in vocab))
        raise ValueError(f"Characters missing from vocabulary: {unknown}")

    words = text.split()
    graphemes = [char for word in words for char in word]

    boundary_positions = set()
    count = 0

    for i, word in enumerate(words):
        if i > 0:
            boundary_positions.add(count)
        count += len(word)

    space_indices = [
        i for i, token in enumerate(tokens)
        if token == "<space>"
    ]

    grapheme_indices = [
        i for i, token in enumerate(tokens)
        if token != "<space>"
    ]

    grid = tgt.io.read_textgrid(str(grid_path))
    tier = grid.get_tier_by_name("phones")

    durations = [0] * len(tokens)
    seen = 0
    start_time = None
    end_time = None
    mismatches = []

    for interval in tier.intervals:
        label = interval.text.strip()
        start = interval.start_time
        end = interval.end_time

        frames = (
            int(np.round(end * SR / HOP))
            - int(np.round(start * SR / HOP))
        )

        if frames < 0:
            raise ValueError("Negative alignment duration")

        if label in ("", "sil", "sp", "spn", "<eps>"):
            if start_time is None or seen >= len(graphemes):
                continue

            if seen in boundary_positions:
                space_idx = next(
                    (
                        idx for idx in space_indices
                        if sum(
                            1 for token in tokens[:idx]
                            if token != "<space>"
                        ) == seen
                    ),
                    None,
                )

                if space_idx is not None:
                    durations[space_idx] += frames

            elif seen > 0:
                durations[grapheme_indices[seen - 1]] += frames

            continue

        if seen >= len(graphemes):
            mismatches.append(("unexpected", label))
            continue

        expected = graphemes[seen]

        if label != expected:
            mismatches.append((expected, label))

        durations[grapheme_indices[seen]] += frames

        if start_time is None:
            start_time = start

        end_time = end
        seen += 1

    if mismatches:
        raise ValueError(f"Sequence mismatch: {mismatches[:3]}")

    if seen != len(graphemes):
        raise ValueError(
            f"Aligned {seen}/{len(graphemes)} graphemes"
        )

    if start_time is None or end_time is None or start_time >= end_time:
        raise ValueError("Invalid alignment boundaries")

    start_frame = int(np.round(start_time * SR / HOP))
    end_frame = int(np.round(end_time * SR / HOP))

    expected_frames = end_frame - start_frame

    if sum(durations) != expected_frames:
        raise ValueError(
            f"Duration sum {sum(durations)} != "
            f"crop frames {expected_frames}"
        )

    wav, original_sr = sf.read(row.audio_path, dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if original_sr != SR:
        raise ValueError(
            f"Expected {SR} Hz audio, got {original_sr}"
        )

    full_mel, full_energy = Audio.tools.get_mel_from_wav(
        wav, stft
    )

    mel = full_mel[:, start_frame:end_frame]
    energy = full_energy[start_frame:end_frame]

    pitch, times = pw.dio(
        wav.astype(np.float64),
        SR,
        frame_period=HOP / SR * 1000,
    )

    pitch = pw.stonemask(
        wav.astype(np.float64),
        pitch,
        times,
        SR,
    )

    pitch = pitch[start_frame:end_frame]

    if mel.shape[1] != expected_frames:
        raise ValueError("Mel frame count mismatch")

    if len(pitch) != expected_frames:
        raise ValueError(
            f"Pitch frames {len(pitch)} != {expected_frames}"
        )

    if len(energy) != expected_frames:
        raise ValueError("Energy frame count mismatch")

    if not (
        np.isfinite(mel).all()
        and np.isfinite(pitch).all()
        and np.isfinite(energy).all()
    ):
        raise ValueError("Non-finite feature values")

    if np.count_nonzero(pitch) <= 1:
        raise ValueError("Insufficient voiced pitch frames")

    if len(durations) != len(tokens):
        raise ValueError("Text-duration length mismatch")

    return {
        "uid": uid,
        "text": text,
        "raw_text": raw_text,
        "tokens": tokens,
        "mel": mel.T.astype(np.float32),
        "pitch": pitch.astype(np.float32),
        "energy": energy.astype(np.float32),
        "duration": np.asarray(durations, dtype=np.int64),
        "pitch_stats": remove_outliers(pitch),
        "energy_stats": remove_outliers(energy),
        "zero_graphemes": sum(
            durations[idx] == 0 for idx in grapheme_indices
        ),
        "zero_spaces": sum(
            durations[idx] == 0 for idx in space_indices
        ),
    }


def save_sample(sample):
    uid = sample["uid"]

    np.save(
        OUT / "mel" / f"{SPEAKER}-mel-{uid}.npy",
        sample["mel"],
    )

    np.save(
        OUT / "pitch" / f"{SPEAKER}-pitch-{uid}.npy",
        sample["pitch"],
    )

    np.save(
        OUT / "energy" / f"{SPEAKER}-energy-{uid}.npy",
        sample["energy"],
    )

    np.save(
        OUT / "duration" / f"{SPEAKER}-duration-{uid}.npy",
        sample["duration"],
    )

    return "|".join([
        uid,
        SPEAKER,
        sample["text"],
        sample["raw_text"],
    ])


def calculate_stats(values):
    values = np.concatenate(values).astype(np.float64)

    stats = [
        float(np.min(values)),
        float(np.max(values)),
        float(np.mean(values)),
        float(np.std(values)),
    ]

    if not np.isfinite(stats).all() or stats[3] <= 0:
        raise ValueError("Invalid feature normalization statistics")

    return stats


def normalize_folder(folder, mean, std):
    for path in folder.glob("*.npy"):
        values = np.load(path).astype(np.float32)
        values = (values - mean) / std
        np.save(path, values.astype(np.float32))


def main():
    for filename in (TRAIN_CSV, VAL_CSV):
        if not filename.exists():
            raise FileNotFoundError(filename)

    with open(
        ROOT / "configs" / "Telugu" / "preprocess.yaml",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    with open(
        DATA / "phone_to_id.json",
        encoding="utf-8",
    ) as f:
        vocab = json.load(f)

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(
            f"{OUT} is not empty. Refusing to overwrite existing data."
        )

    for name in ("mel", "pitch", "energy", "duration"):
        (OUT / name).mkdir(parents=True, exist_ok=True)

    stft = Audio.stft.TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        HOP,
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        SR,
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    train_grids = build_grid_map(TRAIN_GRID)
    val_grids = build_grid_map(VAL_GRID)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_lines = []
    val_lines = []

    pitch_stats = []
    energy_stats = []

    totals = {
        "train_input": len(train_df),
        "val_input": len(val_df),
        "train_saved": 0,
        "val_saved": 0,
        "train_rejected": 0,
        "val_rejected": 0,
        "zero_graphemes": 0,
        "zero_spaces": 0,
    }

    for split, dataframe, grid_map, lines in (
        ("train", train_df, train_grids, train_lines),
        ("val", val_df, val_grids, val_lines),
    ):
        print(f"\nProcessing {split}: {len(dataframe)} utterances")

        for index, row in enumerate(dataframe.itertuples(), start=1):
            uid = str(row.utterance_id)
            grid_path = grid_map.get(uid)

            if grid_path is None:
                totals[f"{split}_rejected"] += 1
                continue

            try:
                sample = process_one(row, grid_path, stft, vocab)
                line = save_sample(sample)

                lines.append(line)

                if split == "train":
                    pitch_stats.append(sample["pitch_stats"])
                    energy_stats.append(sample["energy_stats"])

                totals[f"{split}_saved"] += 1
                totals["zero_graphemes"] += sample["zero_graphemes"]
                totals["zero_spaces"] += sample["zero_spaces"]

            except Exception as exc:
                totals[f"{split}_rejected"] += 1
                print(f"Rejected {uid}: {exc}")

            if index % 100 == 0:
                print(
                    f"{split}: {index}/{len(dataframe)} "
                    f"| saved={totals[f'{split}_saved']} "
                    f"| rejected={totals[f'{split}_rejected']}"
                )

        print(
            f"{split} complete: "
            f"{totals[f'{split}_saved']} saved, "
            f"{totals[f'{split}_rejected']} rejected"
        )

    if not train_lines or not val_lines:
        raise RuntimeError("Training or validation output is empty")

    pitch_summary = calculate_stats(pitch_stats)
    energy_summary = calculate_stats(energy_stats)

    pitch_mean, pitch_std = pitch_summary[2], pitch_summary[3]
    energy_mean, energy_std = energy_summary[2], energy_summary[3]

    normalize_folder(OUT / "pitch", pitch_mean, pitch_std)
    normalize_folder(OUT / "energy", energy_mean, energy_std)

    (OUT / "train.txt").write_text(
        "\n".join(train_lines) + "\n",
        encoding="utf-8",
    )

    (OUT / "val.txt").write_text(
        "\n".join(val_lines) + "\n",
        encoding="utf-8",
    )

    (OUT / "speakers.json").write_text(
        json.dumps({SPEAKER: 0}, indent=2),
        encoding="utf-8",
    )

    stats = {
        "pitch": pitch_summary,
        "energy": energy_summary,
    }

    (OUT / "stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    (OUT / "preprocessing_summary.json").write_text(
        json.dumps(totals, indent=2),
        encoding="utf-8",
    )

    print("\nTELUGU PREPROCESSING SUMMARY")
    print(json.dumps(totals, indent=2))
    print("\nPitch stats:", pitch_summary)
    print("Energy stats:", energy_summary)
    print("Train metadata:", len(train_lines))
    print("Validation metadata:", len(val_lines))
    print("Output:", OUT)


if __name__ == "__main__":
    main()