# Technical Report: FastSpeech 2 for English and Telugu

## 1. Objective

The objective of this project was to implement and evaluate separate FastSpeech 2 text-to-speech (TTS) systems for English and Telugu using the Svarah and IndicTTS Telugu datasets respectively.

The project focused on preparing language-specific datasets, implementing text frontends, extracting acoustic features, training the English model, generating test speech and analysing the duration, pitch and energy predictions of the variance adaptor.

English model training and inference were completed. Telugu data preparation was completed, but Telugu model training and inference could not be completed because a validated Telugu phoneme alignment pipeline was unavailable.

## 2. Reference Implementation and Architecture

**Reference repository:** https://github.com/ming024/FastSpeech2  
**Git commit:** `d4e79eb52e8b01d24703b2dfc0385544092958f3`

The project uses the FastSpeech 2 implementation from the reference repository.

The architecture consists of the following components:

1. **Encoder:** Converts input text or phoneme sequences into hidden representations using Transformer encoder layers.
2. **Variance Adaptor:** Predicts duration, pitch (F0) and energy for the input sequence. These features help the model represent variations in speech.
3. **Length Regulator:** Expands the encoder representations according to predicted or ground-truth phoneme durations to match the length of the mel-spectrogram sequence.
4. **Mel Decoder:** Processes the expanded representations and predicts mel-spectrogram features.
5. **PostNet:** Refines the predicted mel-spectrogram using convolutional layers.
6. **Vocoder:** HiFi-GAN converts the predicted mel-spectrogram into a time-domain waveform.

The selected implementation uses F0-based pitch features rather than the CWT pitch spectrogram variant described in the later FastSpeech 2 paper version. It also includes a Tacotron-2-style PostNet.

For the English experiment, the model configuration was reduced to two encoder layers and two decoder layers, with hidden dimensions of 128. This was done to accommodate the available 4 GB GPU memory.

## 3. Dataset and Split

### 3.1 English — Svarah

**Dataset:** `ai4bharat/Svarah`  
**Source split used:** `test`

The available published dataset split was used to construct task-specific training and validation subsets. Therefore, this is not an official dataset training/validation split.

| Subset | Utterances | Duration |
|---|---:|---:|
| Training | 3,440 | 5.002 hours |
| Validation | 681 | 1.000 hour |
| Total selected | 4,121 | 6.002 hours |

The English dataset was filtered and prepared for acoustic alignment. The original public schema did not expose speaker IDs, so a speaker-independent split could not be guaranteed.

The final split was task-driven and based on the available audio duration and utterance selection. The initially planned shuffle-buffer procedure was not used in the final download procedure; the data was selected sequentially.

### 3.2 Telugu — IndicTTS Telugu

**Dataset:** `SPRINGLab/IndicTTS_Telugu`  
**Source split:** `train`

The Telugu dataset was downloaded and recovered after an interruption during the initial download process.

| Subset | Utterances | Duration |
|---|---:|---:|
| Training | 1,493 | 4.593 hours |
| Validation | 337 | 1.001 hour |
| Total selected | 1,830 | 5.594 hours |

The split was reconstructed deterministically using sorted audio identifiers and transcript metadata.

The final dataset audit confirmed:
- No duplicate audio identifiers.
- No train/validation identifier overlap.
- No missing WAV files in the selected subsets.
- All inspected audio files had a sampling rate of 22,050 Hz.

The dataset includes gender metadata where provided. The Telugu text inventory was constructed from the characters observed in the training transcripts.

The original IndicTTS dataset licence must be checked before redistributing the audio or derived dataset files.

### 3.3 Dataset Preparation Summary

The combined selected dataset contains approximately 11.596 hours of audio across both languages, including training and validation subsets.

The English and Telugu subsets were prepared independently. Their preprocessing and alignment pipelines differ because language-specific phonetic resources are required.

## 4. Preprocessing and Text Frontend

### 4.1 Audio Preprocessing

The audio preprocessing pipeline included:
- Decoding the source audio.
- Converting audio to the required mono waveform representation where necessary.
- Resampling to 22,050 Hz.
- Applying duration and transcript filtering.
- Extracting mel-spectrogram, pitch, energy and duration features where alignment was available.

The acoustic configuration used:
- Sampling rate: 22,050 Hz
- FFT/filter length: 1,024
- Hop length: 256
- Window length: 1,024
- Mel channels: 80
- Mel frequency range: 0–8,000 Hz

### 4.2 English Text Frontend and Alignment

English text was represented using phoneme sequences.

Montreal Forced Aligner (MFA) version 3.4.3 was used with the `english_mfa` acoustic and dictionary models. The matching acoustic and dictionary models were used after an initial dictionary mismatch was identified.

The English alignment audit showed that some utterances contained empty or unknown (`SPN`) phone intervals. To improve alignment quality, utterances with invalid alignments or more than 20% SPN duration were excluded.

| Subset | Original utterances | Retained after filtering | Rejected |
|---|---:|---:|---:|
| Training | 3,440 | 2,738 | 702 |
| Validation | 681 | 538 | 143 |

The preprocessing pipeline successfully generated features for:
- 2,707 English training utterances.
- 535 English validation utterances.

The remaining difference between retained alignments and processed utterances was due to preprocessing exclusions.

The English phoneme inventory was constructed from the aligned TextGrid phone labels. The final vocabulary contained 91 entries, including special tokens.

A custom autocorrelation-based pitch extraction method was used in the English preprocessing pipeline rather than relying on pyworld pitch extraction.

### 4.3 Telugu Text Frontend and Alignment

A Telugu character inventory was constructed from the training transcripts. It contained special tokens, whitespace and 68 observed characters.

This inventory is grapheme-based and is not a validated phoneme inventory.

MFA version 3.4.3 was inspected for available acoustic models. An English acoustic model was available, but a Telugu acoustic model was not available in the installed model collection.

An English acoustic model cannot be treated as a valid Telugu alignment model. Therefore, Telugu phoneme-level alignment and acoustic feature extraction were not completed.

Consequently, the Telugu dataset is prepared, but it is not yet ready for the same phoneme-duration FastSpeech 2 training pipeline used for English.

## 5. Training Configuration

### 5.1 Hardware and Software

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| GPU memory | 4 GB |
| Python | 3.10.11 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| Training language | English |
| Model | FastSpeech 2 |
| Vocoder | HiFi-GAN |

### 5.2 English Model Configuration

| Parameter | Value |
|---|---:|
| Encoder layers | 2 |
| Decoder layers | 2 |
| Encoder hidden dimension | 128 |
| Decoder hidden dimension | 128 |
| Encoder heads | 2 |
| Decoder heads | 2 |
| Variance predictor filter size | 128 |
| Batch size | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | 8 |
| Gradient clipping threshold | 1.0 |
| Warm-up steps | 400 |
| Total optimization steps | 10,000 |
| Checkpoint step | 10,000 |
| Training duration | Approximately 29 minutes |

The model was trained using the optimizer and learning-rate schedule defined in the project configuration. Mixed-precision training was not configured.

The final checkpoint was saved as:

`output/ckpt/English/10000.pth.tar`

Checkpoints were saved at 500-step intervals.

No Telugu training run or Telugu checkpoint was produced.

## 6. Loss Curves and Training Analysis

The English model was trained for 10,000 optimization steps.

The training loss decreased substantially during the initial training phase, followed by fluctuations during the remaining steps. These fluctuations are likely influenced by the small per-step batch size and variation between individual training samples.

The validation loss decreased from 14.5097 at step 100 to 5.1740 at step 10,000.

The minimum recorded validation loss was 5.1552 at step 2,200.

### Final validation losses

| Loss component | Final value |
|---|---:|
| Total loss | 5.1740 |
| Mel loss | 1.1933 |
| Mel PostNet loss | 1.1846 |
| Pitch loss | 1.3844 |
| Energy loss | 1.1103 |
| Duration loss | 0.3014 |

Most validation improvement occurred during the initial training phase. Later validation values remained relatively stable, with some fluctuations.

The final validation loss was close to the minimum recorded value. This suggests that the model reached a relatively stable validation performance, although the curves alone do not establish speech intelligibility or pronunciation quality.

**Figures:**
- `Results/English/training_loss.png`
- `Results/English/validation_loss.png`

No Telugu training or validation loss curves are available because Telugu training was not completed.

## 7. Generated Speech

### 7.1 English

The trained English checkpoint at step 10,000 was used to generate five test samples.

The inference pipeline used an eSpeak NG phonemizer frontend and an approximate mapping from its output to the model's IPA-based training vocabulary.

The five generated WAV files are:

| Sample | Test case | Output file |
|---|---|---|
| E1 | Long and complex word | `E1_antidisestablishmentarianism.wav` |
| E2a | Conduct as a noun | `E2a_conduct_noun.wav` |
| E2b | Conduct as a verb | `E2b_conduct_verb.wav` |
| E3 | Unusual spacing in rhythm | `E3_rhythm.wav` |
| E4 | Time, date and code-switching | `E4_time_date.wav` |

All five files were generated successfully and saved under:

`Results/English/test_generated_audios/`

The inference frontend uses approximate phone mappings for some eSpeak outputs. These mappings are not guaranteed to be phonetically equivalent to the phones used during training. This is an important limitation when interpreting pronunciation results.

A formal listening evaluation was not conducted as part of the recorded experiment. Therefore, no claims about human-rated intelligibility, naturalness or pronunciation accuracy are made.

### 7.2 Telugu

The five Telugu test sentences were identified during project preparation. However, Telugu model training and inference were not completed because a validated Telugu phoneme alignment and duration extraction pipeline was unavailable.

There are no generated Telugu WAV files to report.

## 8. Variance Adaptor Analysis

The English inference script exported predicted duration, pitch and energy values for all five test sentences.

The following figures were generated for each English sample:
- Duration prediction plot.
- Pitch prediction plot.
- Energy prediction plot.

The plots are saved under:

`Results/English/variance_adaptor_plots/`

Each of the five sample folders contains all three plots, giving a total of 15 figures.

The exported values represent inference-time predictions for the five test sentences. They are not ground-truth training-set measurements.

The plots allow examination of how the model allocates predicted durations and varies pitch and energy across input phones. A detailed voiced/unvoiced F0 evaluation and abnormal-value analysis would require additional inspection and should not be inferred from the existence of the plots alone.

No Telugu variance-adaptor plots were generated.

## 9. English–Telugu Comparison

| Aspect | English | Telugu |
|---|---|---|
| Dataset | Svarah | IndicTTS Telugu |
| Source split | Test | Train |
| Training utterances prepared | 3,440 initially; 2,707 processed | 1,493 selected |
| Validation utterances prepared | 681 initially; 535 processed | 337 selected |
| Text representation | IPA-based phone tokens | Character-based inventory |
| Alignment | MFA 3.4.3 with English MFA models | No validated Telugu MFA acoustic model available |
| Acoustic features | Mel, pitch, energy and duration | Not extracted through the complete training pipeline |
| Model training | Completed, 10,000 steps | Not completed |
| Inference | Five WAV files generated | Not completed |
| Variance plots | 15 generated | Not generated |

The English experiment demonstrates a complete training and inference workflow using phoneme-level inputs and aligned duration targets.

The Telugu experiment reached dataset preparation and character inventory construction. However, the character inventory is not equivalent to a phoneme inventory, and a validated alignment method is still required for duration-based FastSpeech 2 training.

The two language pipelines therefore cannot be compared on speech quality or model performance.

## 10. Reproducibility

The following commands were used for the completed English experiment from the project workspace.

### Environment verification

```cmd
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Dataset verification

```cmd
python scripts/verify_data.py
```

### English alignment

MFA version:

```cmd
mfa version
```

English acoustic and dictionary models:

```cmd
mamba run -n aligner mfa model download acoustic english_mfa
mamba run -n aligner mfa model download dictionary english_mfa
```

Alignment was performed on the prepared English training and validation audio/transcript subsets. The filtered alignment manifests were then used for feature extraction.

### English preprocessing

```cmd
python scripts/preprocess_english.py
```

### English training

```cmd
python train.py -p configs/English/preprocess.yaml -m configs/English/model.yaml -t configs/English/train.yaml
```

### English inference

```cmd
python scripts/infer_english.py
```

### Loss plots

```cmd
python scripts/plot_losses.py
```

### Variance-adaptor plots

```cmd
for %F in (Results\English\variance_raw\*.npz) do python scripts\plot_variance.py --input "%F" --out "Results\English\variance_adaptor_plots\%~nF" --sample-id "%~nF"
```

The above commands describe the completed English workflow. Telugu preprocessing, training and inference commands are not reported as completed experiments.

## 11. Limitations and Future Work

The project has the following limitations:

1. **Dataset split:** The Svarah dataset's available published split was used to create task-specific training and validation subsets. It is not an official train/validation split.
2. **Speaker information:** The public Svarah schema used in this experiment did not expose speaker IDs. Speaker-independent splitting could not be verified.
3. **Dataset size:** The selected datasets are relatively small compared with large-scale commercial TTS training corpora.
4. **English alignment:** Some English alignments contained unknown or empty phone intervals. A filtering threshold of 20% SPN duration was used, which is a heuristic and does not guarantee perfect alignment.
5. **English phonemization:** Approximate mappings were required between eSpeak NG output and the training phone vocabulary. This may introduce pronunciation mismatches.
6. **Telugu frontend:** The constructed Telugu inventory is grapheme-based and requires a validated G2P or phoneme frontend before it can be used as a phoneme-level FastSpeech 2 input.
7. **Telugu alignment:** A validated Telugu acoustic alignment model was unavailable in the installed MFA model collection. Telugu duration, pitch and energy feature extraction and model training remain incomplete.
8. **Hardware:** The 4 GB GPU memory constraint required a reduced model configuration, batch size of one and gradient accumulation.
9. **Vocoder:** The HiFi-GAN vocoder was used for English inference. Its compatibility with the exact training data and its contribution to perceived speech quality require further evaluation.
10. **Evaluation:** No formal human listening test, MOS evaluation or objective intelligibility assessment was conducted.

### Future Work

Future work should focus on obtaining a validated Telugu G2P frontend and acoustic alignment method, extracting reliable duration and acoustic features, training the Telugu model and generating the five Telugu test samples. Further English work should evaluate pronunciation mappings, perform listening tests and explore improved alignment and training configurations.

## Conclusion

A resource-constrained English FastSpeech 2 model was trained for 10,000 optimization steps using the Svarah dataset. Five English test utterances were synthesized, and training loss, validation loss and inference-time variance-adaptor plots were generated.

The Telugu dataset was prepared and audited, but Telugu model training was not completed because a validated phoneme-level frontend and alignment pipeline was unavailable.

The project therefore establishes a completed English FastSpeech 2 training and inference workflow and a partially completed Telugu data-preparation workflow. The distinction is important when interpreting the results and assessing the current state of the implementation.