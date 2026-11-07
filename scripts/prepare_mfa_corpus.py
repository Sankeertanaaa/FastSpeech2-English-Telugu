
from pathlib import Path
import pandas as pd
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "English"
OUTPUT_DIR = ROOT / "alignment" / "English"

def clean_transcript(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def prepare_split(csv_name, split_name):
    df = pd.read_csv(DATA_DIR / csv_name)
    corpus_dir = OUTPUT_DIR / split_name / "speaker_1"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []

    for _, row in df.iterrows():
        utterance_id = str(row["utterance_id"])
        audio_path = Path(row["audio_path"])

        if not audio_path.exists():
            missing.append(utterance_id)
            continue

        transcript = clean_transcript(row["text"])
        if not transcript:
            continue

        target_wav = corpus_dir / f"{utterance_id}.wav"
        target_lab = corpus_dir / f"{utterance_id}.lab"

        if not target_wav.exists():
            shutil.copy2(audio_path, target_wav)

        target_lab.write_text(transcript, encoding="utf-8")
        copied += 1

    print(f"{split_name}: {copied} utterances prepared")
    print(f"Missing audio: {len(missing)}")

    if missing:
        print("First missing IDs:", missing[:10])

prepare_split("train.csv", "train")
prepare_split("val.csv", "val")

print("English MFA corpus preparation complete.")
