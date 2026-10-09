# The Effect of Speech Enhancement on Whisper Speech Recognition

**ELEC5305 Individual Project**  
**Name:** Jinyuan Liu  
**Student ID:** 550284873

## Project overview

This project studies whether reducing noise before using Whisper improves speech recognition. Denoising can make speech clearer, but it may also remove useful speech sounds and make recognition worse.

The experiment will compare four approaches:

- No enhancement
- Spectral subtraction
- Wiener filtering
- Pretrained DeepFilterNet

Each approach will process the same noisy recording separately. All recordings will then be tested with the same Whisper model and settings. Clean speech will provide a reference result.

## Experiment plan

| Item | Plan |
| --- | --- |
| Speech data | Mini LibriSpeech recordings with reference transcripts |
| Noise | White noise, DEMAND office noise and another environment, such as traffic |
| Input SNR | 0, 5 and 10 dB |
| Whisper model | `base.en`, without further training |
| First experiment | 10 recordings with office noise at 0 dB |
| Main experiment | 50–100 recordings from several speakers |

A separate small set of recordings will be used to choose parameters. The parameters will then remain unchanged during the final tests. DeepFilterNet will also be used without further training.

## Evaluation

Word Error Rate (WER) will be the main measure. A lower WER means fewer recognition errors. Results will be compared with the original noisy speech to see whether enhancement helps.

SNR, STOI and Whisper's log-Mel features will help explain the results. Spectrograms, transcripts and informal listening will be used to examine selected errors. Processing time will also be recorded.

## Tools and current progress

MATLAB will be used for audio preparation, spectral subtraction, Wiener filtering and signal evaluation. Python will be used for DeepFilterNet, Whisper and WER calculation.

The revised proposal and initial MATLAB script are ready. The script has not yet been tested in MATLAB. DeepFilterNet, Whisper and the evaluation code still need to be added. Experimental results are not yet available.

## Running the initial MATLAB experiment

The experiment starter prepares noisy audio and exports the spectral subtraction and Wiener filtering results. It requires MATLAB and Signal Processing Toolbox.

1. Extract the experiment starter into the project folder.
2. Place ten clean WAV files in `experiment/data/clean/`.
3. Place a single-channel office-noise WAV file at `experiment/data/noise/office.wav`.
4. Add the recording IDs and correct transcripts to `experiment/utterances.csv`.
5. Set MATLAB's Current Folder to `experiment/` and run `matlab/run_pilot.m`.
6. Check the exported audio and the mixing records in `results/mixing_manifest.csv`.

Dataset audio is not included in the starter. Instructions for the Python stage will be added after integration and testing.

## Next steps

Complete the ten-recording experiment, compare WER across the five conditions, add the audio analysis and expand the tests. Code, software versions, experiment records and running instructions will be updated as the project develops.

## Resources

- [Whisper](https://github.com/openai/whisper)
- [Mini LibriSpeech](https://www.openslr.org/31/)
- [DEMAND noise dataset](https://zenodo.org/records/1227121)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [JiWER](https://github.com/jitsi/jiwer)

See the revised proposal for academic references.
