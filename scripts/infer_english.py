
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from phonemizer import phonemize
from phonemizer.separator import Separator
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import FastSpeech2
from utils.model import get_vocoder, vocoder_infer
from utils.tools import plot_mel, expand
from text import load_phone_map

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TESTS = {
    "E1_antidisestablishmentarianism": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a_conduct_noun": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b_conduct_verb": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3_rhythm": "The rhyth m of the orchestra changed unexpectedly.",
    "E4_time_date": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
}

def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_phones(text, vocab):
    raw = phonemize(
        text,
        language="en-gb",
        backend="espeak",
        language_switch="remove-flags",
        strip=True,
        separator=Separator(phone=" ", word=" | ")
    )
    phones = raw.replace("|", "").split()

    mapping = {
        "aɪ": ["aj"],
        "aʊ": ["aw"],
        "eɪ": ["ej"],
        "eə": ["e", "ə"],
        "əʊ": ["ow"],
        "əl": ["ə", "l"],
        "iə": ["ɪ", "ə"],
        "oʊ": ["ow"],
        "ʊɹ": ["ʊ", "ɹ"],
        "ʊə": ["ʊ", "ə"],
        "ɑːɹ": ["ɑː", "ɹ"],
        "ɔːɹ": ["ɔ", "ɹ"],
        "ᵻ": ["ɪ"],
        "ʌ": ["ɐ"],
        "ɔː": ["ɔ"]
    }

    phones = [p for token in phones for p in mapping.get(token, [token])]
    unknown = [p for p in phones if p not in vocab]

    if unknown:
        raise ValueError(f"Unsupported phones: {unknown}")

    return phones

def main():
    preprocess = load_yaml(ROOT / "configs/English/preprocess.yaml")
    model_config = load_yaml(ROOT / "configs/English/model.yaml")
    train_config = load_yaml(ROOT / "configs/English/train.yaml")

    preprocess["path"]["preprocessed_path"] = str(ROOT / "preprocessed_data/English")
    train_config["path"]["ckpt_path"] = str(ROOT / "output/ckpt/English")

    vocab = load_phone_map("English")

    model = FastSpeech2(preprocess, model_config).to(DEVICE)
    checkpoint = torch.load(
        ROOT / "output/ckpt/English/10000.pth.tar",
        map_location=DEVICE,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    vocoder = get_vocoder(model_config, DEVICE)

    output_dir = ROOT / "Results/English/test_generated_audios"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = ROOT / "preprocessed_data/English/stats.json"
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    stats = stats["pitch"] + stats["energy"][:2]

    for name, sentence in TESTS.items():
        phones = make_phones(sentence, vocab)
        ids = torch.tensor(
            [[vocab[p] for p in phones]],
            dtype=torch.long,
            device=DEVICE
        )
        src_lens = torch.tensor([len(phones)], device=DEVICE)
        speakers = torch.tensor([0], device=DEVICE)

        with torch.no_grad():
            prediction = model(
                speakers,
                ids,
                src_lens,
                max(src_lens).item()
            )
            variance_dir = output_dir.parent / "variance_raw"
            variance_dir.mkdir(parents=True, exist_ok=True)

            valid_len = len(phones)

            np.savez_compressed(
                variance_dir / f"{name}.npz",
                pitch=prediction[2][0, :valid_len].detach().cpu().numpy(),
                energy=prediction[3][0, :valid_len].detach().cpu().numpy(),
                log_duration=prediction[4][0, :valid_len].detach().cpu().numpy(),
                duration=prediction[5][0, :valid_len].detach().cpu().numpy(),
            )
            mel_len = prediction[9][0].item()
            mel = prediction[1][0, :mel_len].detach().transpose(0, 1)

            duration = prediction[5][0, :len(phones)].cpu().numpy()
            pitch = prediction[2][0, :len(phones)].cpu().numpy()
            energy = prediction[3][0, :len(phones)].cpu().numpy()

            pitch = expand(pitch, duration)
            energy = expand(energy, duration)

            fig = plot_mel(
                [(mel.cpu().numpy(), pitch, energy)],
                stats,
                [name]
            )
            fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)

            mel_batch = prediction[1][:, :mel_len].transpose(1, 2)
            lengths = prediction[9] * preprocess["preprocessing"]["stft"]["hop_length"]
            wavs = vocoder_infer(
                mel_batch,
                vocoder,
                model_config,
                preprocess,
                lengths=lengths
            )

        import scipy.io.wavfile as wavfile
        wavfile.write(
            str(output_dir / f"{name}.wav"),
            preprocess["preprocessing"]["audio"]["sampling_rate"],
            wavs[0]
        )

        print(f"Generated: {name}.wav | phones={len(phones)} | mel_frames={mel_len}")

    print(f"\nAll outputs saved in: {output_dir}")

if __name__ == "__main__":
    main()
