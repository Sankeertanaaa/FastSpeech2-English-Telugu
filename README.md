# IIIT Hyderabad FastSpeech 2 Assignment Workspace

This is a **dataset-preparation and experiment workspace** for the assignment. It is not a claim that either model has already been trained. No fake checkpoints, audio, loss curves or evaluation results are included.

## Reference
Use the upstream implementation: https://github.com/ming024/FastSpeech2

Keep the upstream license and attribution. Clone it beside this workspace:

```powershell
git clone https://github.com/ming024/FastSpeech2.git third_party/FastSpeech2
```

## Hardware
Target GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM.
Begin with batch size 2 and gradient accumulation 8. Reduce batch size to 1 if needed. Start with a short pilot run before committing to longer training.

## 1. Environment (PowerShell)
Use Python 3.10 or 3.11. Avoid Python 3.13 for the older dependency stack.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build matching your NVIDIA driver, then verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For Svarah, authenticate if Hugging Face requests it:

```powershell
hf auth login
```

## 2. Select approximately 5h train + 1h validation per language

```powershell
python scripts\download_and_split.py --language English --target-root data --train-hours 5 --val-hours 1
python scripts\download_and_split.py --language Telugu --target-root data --train-hours 5 --val-hours 1
```

**Important correction:** Svarah's Hugging Face feature is named `audio_filepath`, not `audio`. The downloader in this workspace handles that field. IndicTTS Telugu uses `audio`.

The script streams the source split, deterministically shuffles with seed 2026, selects validation first and then training, resamples to mono 22.05 kHz WAV and writes CSV manifests. Whole-utterance selection can slightly exceed the requested duration. Streaming does not guarantee only six hours of parquet shards are downloaded.

Dataset facts:
- Svarah: 6,656 examples, about 9.6 hours, only a published `test` split, 117 speakers described in the dataset paper/card. Its HF schema has no speaker ID field. This is a task-driven use of the benchmark for TTS and must be disclosed.
- IndicTTS Telugu: 8,576 rows, about 8.74 hours, 48 kHz audio, two speakers (one male and one female). Its original Indic TTS license terms must be reviewed before public redistribution.

## 3. Verify data and split

```powershell
python scripts\verify_data.py --root data
```

Check the printed hours and counts. Do not continue if either CSV is empty or the script reports overlap/errors.

## 4. Upstream FastSpeech 2 integration — not optional
The files uploaded from the upstream repository are not the entire repository. The full `model/`, `preprocessor/`, `transformer/`, `utils/`, `audio/`, `hifigan/`, `lexicon/` and config directories are needed.

The upstream README uses:
1. `prepare_align.py`
2. Montreal Forced Aligner (MFA) to create TextGrids
3. `preprocess.py` to create mel, pitch, energy and duration features
4. `train.py` to train
5. `synthesize.py` to generate audio

Do not point the Telugu data at the English LJSpeech/MFA setup. The reference implementation supports English and Mandarin, not Telugu. Telugu requires a validated Telugu text frontend/phoneme inventory, lexicon/G2P and a compatible alignment solution. A Telugu training run is not valid until phoneme-duration alignment and feature files have been checked.

See `PATCH_AUDIT.md` for the exact limitations found in the uploaded starter files.

## 5. Exact test utterances
The assignment contains five utterances per language. E2a/E2b and T2a/T2b are separate samples. See `English/test_sentences.txt` and `Telugu/test_sentences.txt`. The English `rhyth m` spelling is retained as shown in the assignment PDF. Telugu words broken across lines in PDF text extraction were checked against the rendered page and joined into the intended words.

## 6. Loss plots
After upstream training has written logs, create loss plots with:

```powershell
python scripts\plot_losses.py --train-log PATH_TO_TRAIN_LOG --val-log PATH_TO_VAL_LOG --out Results\English
```

Repeat for Telugu. These plots must come from actual training logs.

## 7. Variance-adaptor plots
The upstream model's inference output contains pitch, energy and rounded duration predictions. Add the hook in `VARIANCE_EXPORT_HOOK.txt` to the upstream `synthesize.py`, run inference for each test sentence and save the `.npz` outputs. Then plot each one:

```powershell
python scripts\plot_variance.py --input PATH_TO_SAMPLE.npz --out Results\English\variance_adaptor_plots\SAMPLE_ID --sample-id SAMPLE_ID
```

Repeat for each sample and language. These plots must use inference-time predictions, not training targets.

## 8. Report
Use `Report/REPORT_TEMPLATE.md`. Fill it with measured durations, actual training settings, real loss curves, actual audio listening observations and actual variance plots. Do not invent results.

## 9. GitHub
Do not commit raw datasets or large checkpoints directly. Check dataset license terms. Use Git LFS or an external accessible checkpoint link if necessary. Keep the repository public only if dataset and checkpoint licensing permits it.
