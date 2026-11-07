
import re
import unicodedata
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
import tgt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio as Audio

DATA = ROOT / "data" / "Telugu"
SR = 22050
HOP = 256
LIMIT = 50

df = pd.read_csv(DATA / "train.csv").head(LIMIT)

with open(
    ROOT / "configs" / "Telugu" / "preprocess.yaml",
    encoding="utf-8"
) as f:
    config = yaml.safe_load(f)

stft = Audio.stft.TacotronSTFT(
    config["preprocessing"]["stft"]["filter_length"],
    HOP,
    config["preprocessing"]["stft"]["win_length"],
    config["preprocessing"]["mel"]["n_mel_channels"],
    SR,
    config["preprocessing"]["mel"]["mel_fmin"],
    config["preprocessing"]["mel"]["mel_fmax"],
)


def clean_words(text):
    text = str(text)
    text = text.replace("వెళ్ళగొట్టడo", "వెళ్ళగొట్టడం")
    text = text.replace("ఎoత", "ఎంత")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[,.?\"“”‘’!;:]', "", text)
    return re.sub(r"\s+", " ", text).strip().split()


results = []
mismatches = []
errors = []

for row in df.itertuples():
    uid = str(row.utterance_id)

    try:
        words = clean_words(row.text)

        tokens = []
        grapheme_indices = []
        space_indices = {}
        grapheme_count = 0

        for wi, word in enumerate(words):
            if wi > 0:
                space_indices[grapheme_count] = len(tokens)
                tokens.append("<space>")

            for char in word:
                grapheme_indices.append(len(tokens))
                tokens.append(char)
                grapheme_count += 1

        expected = [char for word in words for char in word]
        durations = [0] * len(tokens)

        grid_path = next(
            (DATA / "mfa_train_aligned_fixed").rglob(
                uid + ".TextGrid"
            )
        )

        grid = tgt.io.read_textgrid(str(grid_path))
        tier = grid.get_tier_by_name("phones")

        seen = 0
        start_time = None
        end_time = None
        mismatch = []

        for interval in tier.intervals:
            label = interval.text
            s = interval.start_time
            e = interval.end_time

            start_frame = int(np.round(s * SR / HOP))
            end_frame = int(np.round(e * SR / HOP))
            frames = end_frame - start_frame

            if label in ("", "sil", "sp", "spn"):
                if seen == 0 or seen >= len(expected):
                    continue

                if seen in space_indices:
                    idx = space_indices[seen]
                else:
                    idx = grapheme_indices[seen - 1]

                durations[idx] += frames
                continue

            if seen >= len(expected):
                mismatch.append(("unexpected", label))
                continue

            if label != expected[seen]:
                mismatch.append((expected[seen], label))

            idx = grapheme_indices[seen]
            durations[idx] += frames

            if start_time is None:
                start_time = s

            end_time = e
            seen += 1

        if seen != len(expected):
            mismatch.append(("count", len(expected), seen))

        if mismatch:
            mismatches.append(uid)

        if start_time is None or end_time is None:
            raise ValueError("No valid grapheme intervals found.")

        # Extract mel from the complete original utterance.
        wav, _ = librosa.load(row.audio_path, sr=SR)
        wav = wav.astype(np.float32)

        full_mel, full_energy = Audio.tools.get_mel_from_wav(
            wav, stft
        )

        # Crop mel frames using the MFA boundaries.
        crop_start = int(np.round(start_time * SR / HOP))
        crop_end = int(np.round(end_time * SR / HOP))

        mel = full_mel[:, crop_start:crop_end]
        energy = full_energy[crop_start:crop_end]

        difference = mel.shape[1] - sum(durations)

        zero_graphemes = sum(
            durations[i] == 0 for i in grapheme_indices
        )

        zero_spaces = sum(
            durations[i] == 0 for i in space_indices.values()
        )

        results.append({
            "id": uid,
            "difference": difference,
            "zero_graphemes": zero_graphemes,
            "zero_spaces": zero_spaces,
            "mel_frames": mel.shape[1],
            "duration_sum": sum(durations),
            "sequence_ok": not mismatch,
        })

    except Exception as e:
        errors.append((uid, str(e)))


result_df = pd.DataFrame(results)

print("\n--- TELUGU 50-UTTERANCE AUDIT ---")
print("Requested:", LIMIT)
print("Processed:", len(result_df))
print("Errors:", len(errors))
print("Sequence mismatches:", len(mismatches))

if len(result_df):
    print(
        "Zero-duration graphemes:",
        int(result_df.zero_graphemes.sum())
    )
    print(
        "Zero-duration spaces:",
        int(result_df.zero_spaces.sum())
    )

    print("\nDuration difference distribution:")
    print(
        result_df.difference.value_counts().sort_index()
    )

    print(
        "\nUtterances needing +1 frame:",
        int((result_df.difference == 1).sum())
    )
    print(
        "Utterances needing -1 frame:",
        int((result_df.difference == -1).sum())
    )
    print(
        "Exact matches:",
        int((result_df.difference == 0).sum())
    )
    print(
        "Other differences:",
        int((~result_df.difference.isin([-1, 0, 1])).sum())
    )

    print(
        "\nMean absolute difference:",
        round(result_df.difference.abs().mean(), 3)
    )

result_df.to_csv(
    DATA / "duration_batch_audit.csv",
    index=False
)

print("\nSaved:", DATA / "duration_batch_audit.csv")

if errors:
    print("\nFirst errors:", errors[:5])
