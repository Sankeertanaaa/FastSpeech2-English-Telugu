# FastSpeech 2 for English and Telugu
### IIIT Hyderabad Assignment | Multilingual Text-to-Speech

This repository contains the code, configurations and supporting scripts for an academic project exploring **FastSpeech 2-based Text-to-Speech (TTS) for English and Telugu**.

The project focuses on preparing language-specific datasets, adapting the text and phoneme-processing pipeline, configuring the acoustic model and investigating the challenges involved in building speech synthesis systems for both languages.

The repository documents the implementation workflow, experiments and technical considerations encountered during development. It does not assume successful synthesis or report experimental results that have not been verified.

## Reference Implementation

This project is based on the original [FastSpeech 2 implementation by ming024](https://github.com/ming024/FastSpeech2).

Please retain the original license and attribution when using or modifying upstream code.

For reference, the upstream repository can be cloned separately:

```powershell
git clone https://github.com/ming024/FastSpeech2.git third_party/FastSpeech2
```

## Project Overview

The project explores two language-specific TTS pipelines:

- **English:** Uses the Svarah dataset and an English phoneme-processing pipeline.
- **Telugu:** Uses the IndicTTS Telugu dataset and requires language-specific text processing, phoneme handling and alignment support.

Both pipelines use a FastSpeech 2-based acoustic modelling approach, with HiFi-GAN as the intended neural vocoder.

## Hardware

Development and experimentation were configured around the following system:

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- GPU memory: 4 GB
- Python: 3.10
- PyTorch: CUDA-enabled build

For limited GPU memory, begin with a batch size of 1 or 2 and use gradient accumulation where supported. Run a short pilot experiment before starting longer training sessions.

## 1. Environment Setup

The project uses Python 3.10 or 3.11. Python 3.13 may cause compatibility issues with the older dependency stack.

Open PowerShell in the project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for your system and verify GPU availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If Hugging Face authentication is required for Svarah:

```powershell
hf auth login
```

## 2. Dataset Preparation

The project uses two datasets:

| Language | Dataset | Purpose |
|---|---|---|
| English | Svarah | English speech and text data |
| Telugu | IndicTTS Telugu | Telugu speech and text data |

The intended experimental split is approximately 5 hours for training and 1 hour for validation per language.

Run the dataset preparation scripts:

```powershell
python scripts\download_and_split.py --language English --target-root data --train-hours 5 --val-hours 1

python scripts\download_and_split.py --language Telugu --target-root data --train-hours 5 --val-hours 1
```

The downloader uses the `audio_filepath` field for Svarah and `audio` for IndicTTS Telugu.

The preparation process streams the source split, shuffles examples deterministically using seed 2026, selects validation data before training data, resamples audio to mono 22.05 kHz WAV and generates CSV manifests.

Whole-utterance selection may cause the final duration to exceed the requested target slightly. Streaming also does not guarantee that only the selected six hours of source data are downloaded.

### Dataset Notes

**Svarah**
- 6,656 examples
- Approximately 9.6 hours of audio
- Published `test` split only
- 117 speakers described in the dataset paper/card
- The Hugging Face schema does not provide a speaker ID field

Using this benchmark for TTS is a task-driven adaptation and should be disclosed in the project report.

**IndicTTS Telugu**
- 8,576 rows
- Approximately 8.74 hours of audio
- Original audio sampling rate: 48 kHz
- Two speakers: one male and one female

Review the original IndicTTS license terms before redistributing any dataset material.

## 3. Dataset Verification

After preparing the data, verify the generated files and manifests:

```powershell
python scripts\verify_data.py --root data
```

Check the reported sample counts, durations and validation output. Do not proceed if either manifest is empty or the verification script reports overlap or other data errors.

## 4. FastSpeech 2 Integration

The project builds on the FastSpeech 2 reference implementation. The files in this repository are not a complete copy of the upstream repository, so additional upstream components may be required.

The general FastSpeech 2 workflow consists of:

1. Preparing the speech and text data.
2. Generating phoneme-level alignments using a compatible alignment tool.
3. Extracting mel-spectrogram, pitch, energy and duration features.
4. Training the FastSpeech 2 acoustic model.
5. Generating predicted mel-spectrograms during inference.
6. Using HiFi-GAN to convert mel-spectrograms into speech.

The upstream implementation uses Montreal Forced Aligner (MFA) for alignment and provides preprocessing, training and synthesis scripts.

### Language-specific considerations

**English:** The pipeline uses English phoneme processing and the Svarah dataset. Alignment and phoneme-duration consistency must be verified before training.

**Telugu:** Telugu requires a validated text frontend, phoneme inventory, lexicon or grapheme-to-phoneme approach and a compatible alignment solution. The original reference implementation does not provide a ready-to-use Telugu pipeline.

English alignment resources and phoneme assumptions must not be reused for Telugu without validation.

## 5. Test Utterances

The assignment specifies five test utterances for each language.

The corresponding files are:

- `English/test_sentences.txt`
- `Telugu/test_sentences.txt`

E2a/E2b and T2a/T2b represent separate samples.

The English spelling `rhyth m` is retained as provided in the assignment PDF. Telugu words split across PDF lines were checked against the rendered page to recover their intended forms.

## 6. Training Loss Visualisation

Once training logs are available, the plotting script can be used to generate loss curves:

```powershell
python scripts\plot_losses.py --train-log PATH_TO_TRAIN_LOG --val-log PATH_TO_VAL_LOG --out Results\English
```

Repeat the process for Telugu using its corresponding logs.

Plots should be generated from actual training and validation logs. Do not use illustrative or manually created values as experimental results.

## 7. Variance Adaptor Visualisation

FastSpeech 2 predicts pitch, energy and duration through its variance adaptor.

The inference pipeline can be extended to export these predictions as `.npz` files using the hook documented in `VARIANCE_EXPORT_HOOK.txt`.

After exporting a sample, generate its visualisation using:

```powershell
python scripts\plot_variance.py --input PATH_TO_SAMPLE.npz --out Results\English\variance_adaptor_plots\SAMPLE_ID --sample-id SAMPLE_ID
```

Repeat for each test utterance and language.

These plots should represent inference-time predictions rather than ground-truth training targets.

## 8. Project Report

The report template is available at:

`Report/REPORT_TEMPLATE.md`

The final report should document:

- Dataset preparation and preprocessing
- Actual model and training configurations
- Training and validation loss curves
- Observations from generated speech
- Variance adaptor predictions
- Language-specific challenges
- Limitations and possible improvements

All reported values and observations should be based on actual experiments.

## 9. Repository Structure

The repository is organised around the shared model implementation, language-specific configurations and supporting scripts.

```text
FastSpeech2-English-Telugu/
│
├── configs/
│   ├── English/
│   └── Telugu/
│
├── model/
├── hifigan/
├── preprocessor/
├── text/
├── transformer/
├── utils/
│
├── scripts/
├── dataset.py
├── train.py
├── evaluate.py
├── synthesize.py
├── requirements.txt
└── README.md
```

Dataset files, generated audio, intermediate features and large model checkpoints are excluded from the repository to keep it lightweight.

## 10. Limitations and Reproducibility

This repository is intended to document the implementation and experimental workflow. The presence of training or inference scripts does not, by itself, establish that a model has converged or produces intelligible speech.

For reproducibility:

- Obtain the datasets from their original sources.
- Review the applicable dataset licenses.
- Install the required dependencies.
- Verify preprocessing and alignment outputs before training.
- Use actual checkpoints and logs when reporting results.
- Keep large datasets and model weights outside the Git repository or distribute them through an appropriate storage mechanism.

The original FastSpeech 2 implementation and its license should be acknowledged when reusing upstream components.

---

**Project:** FastSpeech 2 for English and Telugu  
**Institution:** IIIT Hyderabad Assignment  
**Languages:** English and Telugu  
**Architecture:** FastSpeech 2  
**Vocoder:** HiFi-GAN
