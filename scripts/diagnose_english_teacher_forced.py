import sys
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import scipy.io.wavfile as wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import FastSpeech2
from utils.model import get_vocoder, vocoder_infer
from text import load_phone_map

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    preprocess = load_yaml(ROOT / "configs/English/preprocess.yaml")
    model_config = load_yaml(ROOT / "configs/English/model.yaml")

    preprocess["path"]["preprocessed_path"] = str(
        ROOT / "preprocessed_data/English"
    )

    vocab = load_phone_map("English")

    model = FastSpeech2(preprocess, model_config).to(DEVICE)

    checkpoint = torch.load(
        ROOT / "output/ckpt/English/8000.pth.tar",
        map_location=DEVICE,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    vocoder = get_vocoder(model_config, DEVICE)

    basename = "english_1f11ff001417fd05"
    speaker_name = "speaker_1"

    data_dir = ROOT / "preprocessed_data/English"

    phones = [
        "aj", "a", "v", "ʈ", "eː", "k", "ə", "n", "s",
        "ɛ", "v", "a", "ɹ", "a", "ɫ", "t", "ɹ", "i", "p", "s"
    ]

    ids = torch.tensor(
        [[vocab[p] for p in phones]],
        dtype=torch.long,
        device=DEVICE
    )

    mel_gt = np.load(
        data_dir / "mel" / f"{speaker_name}-mel-{basename}.npy"
    )
    pitch_gt = np.load(
        data_dir / "pitch" / f"{speaker_name}-pitch-{basename}.npy"
    )
    energy_gt = np.load(
        data_dir / "energy" / f"{speaker_name}-energy-{basename}.npy"
    )
    duration_gt = np.load(
        data_dir / "duration" / f"{speaker_name}-duration-{basename}.npy"
    )

    print("Phones:", len(phones))
    print("Pitch:", len(pitch_gt))
    print("Energy:", len(energy_gt))
    print("Durations:", len(duration_gt))
    print("Ground-truth mel:", mel_gt.shape)

    assert len(phones) == len(pitch_gt) == len(energy_gt) == len(duration_gt)

    speakers = torch.tensor([0], dtype=torch.long, device=DEVICE)
    src_lens = torch.tensor([len(phones)], device=DEVICE)

    pitch = torch.tensor(pitch_gt, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    energy = torch.tensor(energy_gt, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    duration = torch.tensor(duration_gt, dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        mel_len_gt = torch.tensor([int(duration.sum().item())], device=DEVICE)

        prediction = model(
            speakers,
            ids,
            src_lens,
            len(phones),
            mel_lens=mel_len_gt,
            max_mel_len=mel_len_gt.item(),
            p_targets=pitch,
            e_targets=energy,
            d_targets=duration
        )

        output_dir = ROOT / "Results/English/teacher_forced_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        mel_len = prediction[9][0].item()
        np.save(output_dir / "raw_decoder_mel.npy", prediction[0][0, :mel_len].cpu().numpy())
        np.save(output_dir / "postnet_mel.npy", prediction[1][0, :mel_len].cpu().numpy())
        np.save(output_dir / "teacher_forced_mel.npy", prediction[1][0, :mel_len].detach().cpu().numpy())
        mel = prediction[1][:, :mel_len].transpose(1, 2)

        wavs = vocoder_infer(
            mel,
            vocoder,
            model_config,
            preprocess
        )

    output_dir = ROOT / "Results/English/teacher_forced_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavfile.write(
        str(output_dir / "teacher_forced.wav"),
        preprocess["preprocessing"]["audio"]["sampling_rate"],
        wavs[0]
    )

    print("Generated:", output_dir / "teacher_forced.wav")
    print("Mel frames:", mel_len)


if __name__ == "__main__":
    main()