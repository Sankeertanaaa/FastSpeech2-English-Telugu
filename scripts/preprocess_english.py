
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import get_window
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import audio as Audio


LANGUAGE = "English"

RAW_DIR = ROOT / "data" / LANGUAGE / "raw_for_fastspeech"
DATA_DIR = ROOT / "data" / LANGUAGE
ALIGN_DIR = ROOT / "alignment" / LANGUAGE
OUTPUT_DIR = ROOT / "preprocessed_data" / LANGUAGE

SPLITS = {
    "train": {
        "csv": DATA_DIR / "train_aligned_filtered.csv",
        "alignment": ALIGN_DIR / "aligned_train" / "speaker_1",
    },
    "val": {
        "csv": DATA_DIR / "val_aligned_filtered.csv",
        "alignment": ALIGN_DIR / "aligned_val" / "speaker_1",
    },
}


def extract_pitch_autocorrelation(wav, sr, hop_length, frame_length=1024):
    """
    Lightweight autocorrelation pitch estimator.
    Returns one pitch value per hop-length frame.
    Unvoiced frames are represented by zero.
    """

    wav = np.asarray(wav, dtype=np.float32)

    if wav.size == 0:
        return np.zeros(0, dtype=np.float32)

    window = get_window("hann", frame_length, fftbins=True).astype(
        np.float32
    )

    min_lag = max(1, int(sr / 600))
    max_lag = min(int(sr / 50), frame_length - 1)

    pitches = []

    for start in range(0, len(wav), hop_length):
        frame = wav[start:start + frame_length]

        if len(frame) < frame_length:
            frame = np.pad(
                frame,
                (0, frame_length - len(frame))
            )

        frame = frame.astype(np.float32)
        frame -= np.mean(frame)

        rms = np.sqrt(np.mean(frame ** 2))

        if rms < 0.01:
            pitches.append(0.0)
            continue

        frame *= window

        correlation = np.correlate(
            frame,
            frame,
            mode="full"
        )[frame_length - 1:]

        if len(correlation) <= max_lag:
            pitches.append(0.0)
            continue

        section = correlation[min_lag:max_lag + 1]

        if len(section) == 0:
            pitches.append(0.0)
            continue

        lag = int(np.argmax(section)) + min_lag

        if lag <= 0:
            pitches.append(0.0)
            continue

        frequency = sr / lag

        if 50 <= frequency <= 600:
            pitches.append(float(frequency))
        else:
            pitches.append(0.0)

    return np.asarray(pitches, dtype=np.float32)


def remove_outliers(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) < 4:
        return values

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = values[
        (values > lower) & (values < upper)
    ]

    return filtered if len(filtered) else values


def get_alignment(tier, sampling_rate, hop_length):
    silence = {"sil", "sp", "spn"}

    phones = []
    durations = []

    start_time = None
    end_time = None

    for interval in tier.intervals:
        phone = interval.text.strip()

        if not phone:
            continue

        if start_time is None:
            if phone in silence:
                continue

            start_time = interval.start_time

        phones.append(phone)

        start_frame = round(
            interval.start_time * sampling_rate / hop_length
        )

        end_frame = round(
            interval.end_time * sampling_rate / hop_length
        )

        durations.append(
            max(0, int(end_frame - start_frame))
        )

        end_time = interval.end_time

    while phones and phones[-1] in silence:
        phones.pop()
        durations.pop()

    if not phones or start_time is None or end_time is None:
        return None

    return phones, durations, start_time, end_time


def process_utterance(row, alignment_folder, stft, config):

    uid = str(row["utterance_id"])

    wav_path = RAW_DIR / "speaker_1" / f"{uid}.wav"
    lab_path = RAW_DIR / "speaker_1" / f"{uid}.lab"
    tg_path = alignment_folder / f"{uid}.TextGrid"

    if not wav_path.exists():
        return None

    if not lab_path.exists() or not tg_path.exists():
        return None

    textgrid = Audio  # Keep audio module separate from TextGrid parsing.
    import tgt

    tg = tgt.io.read_textgrid(str(tg_path))
    tier = tg.get_tier_by_name("phones")

    if tier is None:
        return None

    sr = config["preprocessing"]["audio"]["sampling_rate"]
    hop = config["preprocessing"]["stft"]["hop_length"]

    alignment = get_alignment(tier, sr, hop)

    if alignment is None:
        return None

    phones, durations, start, end = alignment

    total_frames = sum(durations)

    if total_frames <= 1:
        return None

    wav, actual_sr = librosa.load(
        str(wav_path),
        sr=sr,
        mono=True
    )

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    wav = wav[start_sample:end_sample].astype(np.float32)

    if len(wav) < 512:
        return None

    # Generate mel spectrogram and frame-level energy.
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)

    with torch.no_grad():
        mel, energy = stft.mel_spectrogram(wav_tensor)

    mel = mel.squeeze(0).cpu().numpy().T
    energy = energy.squeeze(0).cpu().numpy()

    mel = mel[:total_frames]
    energy = energy[:total_frames]

    if len(mel) != total_frames or len(energy) != total_frames:
        return None

    # Extract frame-level pitch using autocorrelation.
    pitch = extract_pitch_autocorrelation(
        wav,
        sr,
        hop,
        frame_length=1024
    )

    pitch = pitch[:total_frames]

    if len(pitch) < total_frames:
        pitch = np.pad(
            pitch,
            (0, total_frames - len(pitch))
        )

    if np.count_nonzero(pitch) <= 1:
        return None

    # Interpolate unvoiced pitch frames before phoneme averaging.
    voiced = np.where(pitch > 0)[0]

    if len(voiced) < 2:
        return None

    pitch = np.interp(
        np.arange(total_frames),
        voiced,
        pitch[voiced]
    ).astype(np.float32)

    # Convert frame-level pitch and energy to phoneme-level features.
    phone_pitch = []
    phone_energy = []

    position = 0

    for duration in durations:

        next_position = position + duration

        if duration > 0:
            phone_pitch.append(
                float(np.mean(pitch[position:next_position]))
            )

            phone_energy.append(
                float(np.mean(energy[position:next_position]))
            )
        else:
            phone_pitch.append(0.0)
            phone_energy.append(0.0)

        position = next_position

    phone_pitch = np.asarray(phone_pitch, dtype=np.float32)
    phone_energy = np.asarray(phone_energy, dtype=np.float32)

    if len(phone_pitch) != len(phones):
        return None

    if len(phone_energy) != len(phones):
        return None

    raw_text = lab_path.read_text(
        encoding="utf-8"
    ).strip()

    metadata = "|".join([
        uid,
        "speaker_1",
        " ".join(phones),
        raw_text
    ])

    return {
        "metadata": metadata,
        "mel": mel.astype(np.float32),
        "duration": np.asarray(durations, dtype=np.int64),
        "pitch": phone_pitch,
        "energy": phone_energy,
    }


def main():

    config_path = ROOT / "configs" / "English" / "preprocess.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sr = config["preprocessing"]["audio"]["sampling_rate"]

    stft = Audio.stft.TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        sr,
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    stft.eval()

    feature_folders = [
        "mel",
        "pitch",
        "energy",
        "duration"
    ]

    for folder in feature_folders:
        (OUTPUT_DIR / folder).mkdir(
            parents=True,
            exist_ok=True
        )

    results = {}

    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()

    for split, paths in SPLITS.items():

        df = pd.read_csv(paths["csv"])

        processed = []

        print(
            f"\nProcessing {split}: {len(df)} utterances"
        )

        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            result = process_utterance(
                row,
                paths["alignment"],
                stft,
                config
            )

            if result is None:
                continue

            uid = str(row["utterance_id"])

            np.save(
                OUTPUT_DIR / "mel" / f"speaker_1-mel-{uid}.npy",
                result["mel"]
            )

            np.save(
                OUTPUT_DIR / "duration" / f"speaker_1-duration-{uid}.npy",
                result["duration"]
            )

            np.save(
                OUTPUT_DIR / "pitch" / f"speaker_1-pitch-{uid}.npy",
                result["pitch"]
            )

            np.save(
                OUTPUT_DIR / "energy" / f"speaker_1-energy-{uid}.npy",
                result["energy"]
            )

            if split == "train":

                p = remove_outliers(result["pitch"])
                e = remove_outliers(result["energy"])

                if len(p):
                    pitch_scaler.partial_fit(
                        p.reshape(-1, 1)
                    )

                if len(e):
                    energy_scaler.partial_fit(
                        e.reshape(-1, 1)
                    )

            processed.append(result["metadata"])

        results[split] = processed

        print(
            f"{split} successfully processed: {len(processed)}"
        )

    if not results["train"]:
        raise RuntimeError(
            "No training utterances were successfully processed."
        )

    if not hasattr(pitch_scaler, "mean_"):
        raise RuntimeError(
            "Unable to calculate training pitch statistics."
        )

    if not hasattr(energy_scaler, "mean_"):
        raise RuntimeError(
            "Unable to calculate training energy statistics."
        )

    pitch_mean = float(pitch_scaler.mean_[0])
    pitch_std = max(float(pitch_scaler.scale_[0]), 1e-8)

    energy_mean = float(energy_scaler.mean_[0])
    energy_std = max(float(energy_scaler.scale_[0]), 1e-8)

    normalized_pitch = []
    normalized_energy = []

    # Normalize train and validation using training statistics.
    for split in ["train", "val"]:

        for line in results[split]:

            uid = line.split("|", 1)[0]

            pitch_path = (
                OUTPUT_DIR / "pitch" /
                f"speaker_1-pitch-{uid}.npy"
            )

            energy_path = (
                OUTPUT_DIR / "energy" /
                f"speaker_1-energy-{uid}.npy"
            )

            pitch = np.load(pitch_path)
            energy = np.load(energy_path)

            pitch = (
                pitch - pitch_mean
            ) / pitch_std

            energy = (
                energy - energy_mean
            ) / energy_std

            np.save(
                pitch_path,
                pitch.astype(np.float32)
            )

            np.save(
                energy_path,
                energy.astype(np.float32)
            )

            if split == "train":
                normalized_pitch.extend(
                    pitch.tolist()
                )

                normalized_energy.extend(
                    energy.tolist()
                )

    stats = {
        "pitch": [
            float(np.min(normalized_pitch)),
            float(np.max(normalized_pitch)),
            pitch_mean,
            pitch_std
        ],
        "energy": [
            float(np.min(normalized_energy)),
            float(np.max(normalized_energy)),
            energy_mean,
            energy_std
        ]
    }

    (OUTPUT_DIR / "stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8"
    )

    (OUTPUT_DIR / "speakers.json").write_text(
        json.dumps(
            {"speaker_1": 0},
            indent=2
        ),
        encoding="utf-8"
    )

    for split in ["train", "val"]:

        metadata_file = OUTPUT_DIR / f"{split}.txt"

        metadata_file.write_text(
            "\n".join(results[split]) + "\n",
            encoding="utf-8"
        )

    print("\nEnglish preprocessing complete.")

    print(
        "Training utterances:",
        len(results["train"])
    )

    print(
        "Validation utterances:",
        len(results["val"])
    )

    print(
        "Pitch statistics:",
        stats["pitch"]
    )

    print(
        "Energy statistics:",
        stats["energy"]
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
