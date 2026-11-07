# FastSpeech 2 for English and Telugu

**IIIT Hyderabad Academic Project | Multilingual Text-to-Speech**

This repository contains the code, configuration files and supporting scripts for exploring FastSpeech 2-based Text-to-Speech (TTS) systems for English and Telugu.

The project focuses on preparing language-specific datasets, adapting the text-processing pipeline and configuring the FastSpeech 2 architecture for multilingual speech synthesis. It also includes utilities for dataset verification, alignment preparation and experiment analysis.

The repository documents the implementation and experimentation workflow. The presence of scripts or configurations does not necessarily indicate that the corresponding training, synthesis or evaluation stages have been completed. Results will be reported only when supported by actual experiments.

## Project Overview

The project explores two language-specific TTS pipelines:

- **English:** Uses the Svarah dataset.
- **Telugu:** Uses the IndicTTS Telugu dataset.

FastSpeech 2 is used as the acoustic model architecture, with HiFi-GAN included as the intended vocoder component.

The work involves dataset preparation, text and phoneme processing, alignment, model configuration and subsequent experimentation.

## Repository Structure

```text
FastSpeech2-English-Telugu/
│
├── configs/
│   ├── English/
│   ├── Telugu/
│   ├── English.yaml
│   └── Telugu.yaml
│
├── model/
├── preprocessor/
├── text/
├── transformer/
├── utils/
│
├── hifigan/
│
├── scripts/
│   ├── download_and_split.py
│   ├── verify_data.py
│   ├── audit_alignments.py
│   ├── prepare_mfa_corpus.py
│   └── plot_variance.py
│
├── English/
│   └── test_sentences.txt
│
├── Telugu/
│   └── test_sentences.txt
│
├── Report/
│   └── REPORT_TEMPLATE.md
│
├── VARIANCE_EXPORT_HOOK.txt
├── dataset.py
├── train.py
├── evaluate.py
├── synthesize.py
├── requirements.txt
└── README.md
```

## Datasets

The project uses the following datasets:

| Language | Dataset | Description |
|---|---|---|
| English | Svarah | English speech dataset |
| Telugu | IndicTTS Telugu | Telugu speech dataset |

Dataset preparation and verification utilities are included in the `scripts/` directory.

**Note:** Dataset downloads, preprocessing and alignment generation should be performed separately. Raw datasets and generated intermediate files are not included in this repository.

## Available Scripts

| Script | Purpose |
|---|---|
| `download_and_split.py` | Dataset download and splitting workflow |
| `verify_data.py` | Dataset verification |
| `audit_alignments.py` | Alignment inspection and auditing |
| `prepare_mfa_corpus.py` | Preparation of corpus files for Montreal Forced Aligner |
| `plot_variance.py` | Variance-related visualization utility |

Additional model and training components are organized into their respective directories.

## Model Architecture

FastSpeech 2 is a non-autoregressive neural TTS architecture. Its main components include:

- Text and phoneme embedding
- Transformer-based encoder
- Variance adaptor
- Transformer-based decoder
- Mel-spectrogram generation

HiFi-GAN is included as the vocoder component intended for converting generated mel-spectrograms into audio.

## Language-Specific Considerations

English and Telugu require separate text-processing configurations.

The Telugu pipeline requires particular attention to Unicode text handling, phoneme representation, pronunciation processing and alignment compatibility.

The language-specific configuration files are maintained under `configs/`.

## Environment Setup

The project uses Python and PyTorch.

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Ensure that the installed PyTorch version is compatible with the available CUDA environment if GPU acceleration is required.

## Training and Experimentation

The repository includes the core model implementation, dataset utilities and language-specific configurations required for the project workflow.

The general workflow involves:

1. Preparing and verifying the datasets.
2. Processing text and phoneme representations.
3. Preparing and validating alignments.
4. Configuring the model for the respective language.
5. Running model training.
6. Evaluating generated speech and inspecting results.

These steps describe the intended workflow and should not be interpreted as confirmation that every stage has been completed.

## Analysis and Documentation

The repository also includes:

- Test sentences for English and Telugu.
- Alignment auditing utilities.
- Variance visualization utilities.
- A report template for documenting methodology, experiments and observations.

Training metrics, generated audio, evaluation results and model checkpoints should be added only when they are available and verified.

## Acknowledgements

This project builds upon the FastSpeech 2 implementation by [ming024](https://github.com/ming024/FastSpeech2).

The datasets and supporting resources are acknowledged to their respective authors and maintainers. Please refer to their original documentation and licenses before use.

---

**Project:** FastSpeech 2 for English and Telugu  
**Institution:** IIIT Hyderabad  
**Domain:** Speech Processing | Deep Learning | Text-to-Speech  
**Architecture:** FastSpeech 2  
**Vocoder:** HiFi-GAN
