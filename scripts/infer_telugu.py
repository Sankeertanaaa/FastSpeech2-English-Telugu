
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
import yaml
import scipy.io.wavfile as wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import FastSpeech2
from utils.model import get_vocoder, vocoder_infer
from utils.tools import plot_mel, expand
from text import load_phone_map

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TESTS = {
    "T1_named_entities": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a_pattu_grip": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b_pattu_silk": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3_tiger": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
    "T4_time_date": "ఈ సమావేశం రెండు వేల ఇరవై ఆరు నవంబర్ పన్నెండవ తేదీ ఉదయం పది గంటల నలభై ఐదు నిమిషాలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
}


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(
        r"(?<=[\u0C00-\u0C7F])o(?=[\u0C00-\u0C7F\s,?.!;:]|$)",
        "ం",
        text
    )
    text = re.sub(r'[,.?"“”‘’!;:]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_tokens(text, vocab):
    text = clean_text(text)

    tokens = [
        "<space>" if char.isspace() else char
        for char in text
    ]

    unknown = sorted(set(token for token in tokens if token not in vocab))

    if unknown:
        raise ValueError(
            f"Unsupported Telugu characters in input: {unknown}"
        )

    return tokens


def main():
    preprocess = load_yaml(ROOT / "configs/Telugu/preprocess.yaml")
    model_config = load_yaml(ROOT / "configs/Telugu/model.yaml")
    train_config = load_yaml(ROOT / "configs/Telugu/train.yaml")

    preprocess["path"]["preprocessed_path"] = str(
        ROOT / "preprocessed_data/Telugu"
    )

    train_config["path"]["ckpt_path"] = str(
        ROOT / "output/ckpt/Telugu"
    )

    vocab = load_phone_map("Telugu")

    model = FastSpeech2(preprocess, model_config).to(DEVICE)

    checkpoint = torch.load(
        ROOT / "output/ckpt/Telugu/10000.pth.tar",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    vocoder = get_vocoder(model_config, DEVICE)

    output_dir = ROOT / "Results/Telugu/test_generated_audios"
    output_dir.mkdir(parents=True, exist_ok=True)

    variance_dir = ROOT / "Results/Telugu/variance_raw"
    variance_dir.mkdir(parents=True, exist_ok=True)

    with open(
        ROOT / "preprocessed_data/Telugu/stats.json",
        encoding="utf-8"
    ) as f:
        stats = json.load(f)

    stats = stats["pitch"] + stats["energy"][:2]

    for name, sentence in TESTS.items():

        cleaned = clean_text(sentence)
        tokens = make_tokens(sentence, vocab)

        ids = torch.tensor(
            [[vocab[token] for token in tokens]],
            dtype=torch.long,
            device=DEVICE
        )

        src_lens = torch.tensor(
            [len(tokens)],
            dtype=torch.long,
            device=DEVICE
        )

        speakers = torch.tensor([0], device=DEVICE)

        with torch.no_grad():
            prediction = model(
                speakers,
                ids,
                src_lens,
                len(tokens)
            )

            valid_len = len(tokens)

            np.savez_compressed(
                variance_dir / f"{name}.npz",
                pitch=prediction[2][0, :valid_len].cpu().numpy(),
                energy=prediction[3][0, :valid_len].cpu().numpy(),
                log_duration=prediction[4][0, :valid_len].cpu().numpy(),
                duration=prediction[5][0, :valid_len].cpu().numpy()
            )

            mel_len = prediction[9][0].item()

            mel = prediction[1][0, :mel_len].detach().transpose(0, 1)

            duration = prediction[5][0, :valid_len].cpu().numpy()
            pitch = prediction[2][0, :valid_len].cpu().numpy()
            energy = prediction[3][0, :valid_len].cpu().numpy()

            pitch_expanded = expand(pitch, duration)
            energy_expanded = expand(energy, duration)

            fig = plot_mel(
                [(mel.cpu().numpy(), pitch_expanded, energy_expanded)],
                stats,
                [name]
            )

            fig.savefig(
                output_dir / f"{name}.png",
                bbox_inches="tight"
            )

            import matplotlib.pyplot as plt
            plt.close(fig)

            mel_batch = prediction[1][:, :mel_len].transpose(1, 2)

            lengths = (
                prediction[9]
                * preprocess["preprocessing"]["stft"]["hop_length"]
            )

            wavs = vocoder_infer(
                mel_batch,
                vocoder,
                model_config,
                preprocess,
                lengths=lengths
            )

        wavfile.write(
            str(output_dir / f"{name}.wav"),
            preprocess["preprocessing"]["audio"]["sampling_rate"],
            wavs[0]
        )

        print(
            f"Generated: {name}.wav | "
            f"graphemes={len(tokens)} | mel_frames={mel_len}"
        )

    print("\nAll Telugu outputs saved in:", output_dir)


if __name__ == "__main__":
    main()
