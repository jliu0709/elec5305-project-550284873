# The Effect of Speech Enhancement on Whisper Speech Recognition

This ELEC5305 project studies whether speech enhancement improves the recognition performance of a fixed Whisper model. The experiment compares:

- no enhancement;
- spectral subtraction;
- Wiener filtering;
- pretrained DeepFilterNet3;
- clean speech as a reference.

No new neural network was trained. The main question is whether a cleaner signal also gives a lower word error rate (WER).

## Main experiment files

The final experiment was run locally with MATLAB and VS Code. Google Colab is not required.

- `ELEC5305_Final_MATLAB.mlx` contains the complete MATLAB stage. It prepares the mixtures, runs spectral subtraction and Wiener filtering, evaluates waveform metrics, and creates MATLAB figures.
- `ELEC5305_Final_VSCode.ipynb` contains the complete Python stage. It runs DeepFilterNet3, Whisper, WER evaluation, log-Mel analysis, plots, and result validation.

The two files should be run in this order:

1. Run `ELEC5305_Final_MATLAB.mlx` in MATLAB.
2. Run `ELEC5305_Final_VSCode.ipynb` in VS Code with a Python 3.12 kernel.

The earlier pilot experiment is stored separately and is not required to rerun this final experiment.

## Experiment design

- Pilot: 10 Mini LibriSpeech utterances, office noise, 0 dB.
- Development set: 5 utterances from 5 speakers, used only for parameter selection.
- Test set: 50 utterances from 21 speakers.
- Development and test speakers are different.
- Noise types: white noise, DEMAND office noise, and DEMAND traffic noise.
- Input SNRs: 0, 5, and 10 dB.
- Whisper model: `base.en` with fixed decoding settings.
- Main metrics: corpus WER, substitutions, deletions, insertions, STOI, output SNR, SNR improvement, log-Mel distance, and real-time factor.

For the four main processing conditions, the test contains 1,800 noisy or enhanced recognition cases. It also contains 50 clean recognitions and 450 results for the modified spectral-subtraction setting.

## Fair comparison

All base speech is mono 16 kHz. For the same utterance, noise type, and SNR, every method receives the same saved noisy signal.

The noise gain is:

`noise_gain = sqrt(P_clean / (P_noise * 10^(SNR_dB/10)))`

A common scale is applied to the mixture and the clean evaluation reference to prevent clipping without changing the target SNR. A one-second noise-only prefix is used for the initial noise estimate. This prefix is removed before recognition and evaluation. Clean speech is not used by any enhancement method.

Spectral subtraction and Wiener filtering use the same settings:

- 512-sample frame;
- 128-sample hop;
- square-root periodic Hann window;
- the same initial noise estimate;
- the noisy phase for reconstruction.

DeepFilterNet3 uses the official DeepFilterNet v0.5.6 model. Audio is resampled from 16 kHz to 48 kHz and then back to 16 kHz. Delay compensation is enabled.

## Final results

The main test results are:

| Method | WER (%) | Difference from no enhancement (percentage points) |
| --- | ---: | ---: |
| No enhancement | 10.54 | 0.00 |
| Spectral subtraction, alpha = 1.0 | 10.24 | -0.30 |
| Wiener filtering | 11.03 | +0.49 |
| DeepFilterNet3 | 14.28 | +3.74 |
| Spectral subtraction, alpha = 1.5 | 10.62 | +0.08 |

Clean-speech WER was 5.02%. Spectral subtraction with `alpha = 1.0` had the lowest overall noisy-speech WER, but its 95% confidence interval included zero. DeepFilterNet3 improved output SNR and usually improved STOI, but it increased WER. This result shows that better signal metrics do not always mean better Whisper recognition.

The development set selected `alpha = 1.5`. On the independent test set, its WER was 10.62%, compared with 10.24% for the original `alpha = 1.0` setting.

## Folder structure

The complete final experiment folder uses this structure:

```text
ELEC5305_Final/
  ELEC5305_Final_MATLAB.mlx
  ELEC5305_Final_VSCode.ipynb
  data/
    clean/development/
    clean/test/
    noise/
    splits/
  results/
    final/development/
    final/test/
  vscode_code/
  data.zip
  ELEC5305_Final_Results.zip
```

## Data

The experiment uses:

- LibriSpeech speech and transcripts from [OpenSLR 12](https://www.openslr.org/12);
- office and traffic noise from [DEMAND](https://zenodo.org/records/1227121);
- generated white noise.

The exact development and test lists are stored in:

- `data/splits/development_manifest.csv`;
- `data/splits/test_manifest.csv`.

The final experiment folder already contains the prepared speech, noise files, manifests, and saved results used in the report.

## Software

### MATLAB

- MATLAB R2025b;
- Signal Processing Toolbox;
- Audio Toolbox;
- Parallel Computing Toolbox is optional.

### Python

- Python 3.12;
- `openai-whisper 20250625`;
- `torch 2.9.1+cu128`;
- CUDA 12.8 for the completed GPU run;
- `jiwer 4.0.0`;
- NumPy, pandas, SciPy, SoundFile, Matplotlib, pystoi, and imageio-ffmpeg.

The completed run used an NVIDIA GeForce RTX 5070 Laptop GPU. A CUDA GPU reduces runtime, but Whisper can also run on CPU.

Example environment setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install openai-whisper==20250625 jiwer==4.0.0 scipy pandas soundfile matplotlib pystoi imageio-ffmpeg
```

## Running the final experiment

### 1. MATLAB

Open the final experiment folder in MATLAB and run:

`ELEC5305_Final_MATLAB.mlx`

The MATLAB stage creates the saved noisy mixtures and the classical enhancement outputs. It also calculates waveform metrics and produces the MATLAB figures.

### 2. VS Code

Open this file in VS Code:

`ELEC5305_Final_VSCode.ipynb`

Select a Python 3.12 kernel. At the start of the notebook, change `PROJECT_ROOT` to the local final experiment folder. If the folder is moved to another location, also update the notebook's `%%writefile` paths before running all cells.

The notebook then:

1. installs or checks the Python packages;
2. prepares the Python scripts;
3. runs DeepFilterNet3;
4. runs fixed Whisper `base.en`;
5. selects the spectral-subtraction coefficient using the development set;
6. evaluates the test set;
7. calculates log-Mel distance and waveform metrics;
8. creates the final figures;
9. validates the saved results.

The long stages save intermediate files, so the experiment can be restarted without repeating every completed item.

## Important result files

- `results/final/validation_report.json`: experiment completeness and alignment checks.
- `results/final/test/asr/wer_summary_overall.csv`: overall WER and error counts.
- `results/final/test/asr/wer_summary_by_condition.csv`: WER for each noise and SNR.
- `results/final/test/metrics_summary.csv`: SNR, STOI, alignment, and processing time.
- `results/final/test/logmel_summary.csv`: Whisper log-Mel distances.
- `results/final/test/delta_wer_cluster_bootstrap.csv`: paired confidence intervals.
- `results/final/test/figures/`: final figures used in the report.
- `results/final/test/examples/`: selected success and failure cases.

Corpus WER is calculated as total `(substitutions + deletions + insertions) / reference words`. It is not the simple average of utterance WERs. Negative delta WER means the enhancement method improved recognition relative to the matched noisy condition.

## Third-party resources

- LibriSpeech / OpenSLR speech and transcripts;
- DEMAND environmental noise;
- OpenAI Whisper pretrained `base.en` model;
- DeepFilterNet pretrained DeepFilterNet3 model and official inference program;
- jiwer for WER alignment and error counts;
- pystoi for STOI.

This project did not train a new neural network. The main work was preparing noisy speech, implementing spectral subtraction and Wiener filtering, using pretrained DeepFilterNet3 and Whisper, calculating the evaluation metrics, plotting the results, and analysing cases where better signal quality did not lead to better recognition.
