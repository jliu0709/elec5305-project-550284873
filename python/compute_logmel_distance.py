"""Compute Whisper log-Mel distances for the pilot audio."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisper


CONDITIONS = ["clean", "noisy", "spectral_subtraction", "wiener", "deepfilter"]


def read_mono_16k(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != whisper.audio.SAMPLE_RATE:
        raise ValueError(f"Expected 16 kHz audio: {path}")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite samples: {path}")
    return audio


def official_logmel(audio: np.ndarray) -> torch.Tensor:
    """Use Whisper's normal 30 s input and 80-bin log-Mel features."""
    padded = whisper.pad_or_trim(audio)
    return whisper.log_mel_spectrogram(padded, n_mels=80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    audio_dir = project_root / "results" / "audio"
    analysis_dir = project_root / "results" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with (project_root / "utterances.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 10:
        raise ValueError("The pilot manifest must contain exactly 10 utterances.")

    rows: list[dict[str, float | int | str]] = []
    for item in manifest:
        utterance_id = item["utterance_id"]
        clean = read_mono_16k(audio_dir / f"{utterance_id}_clean.wav")
        clean_mel = official_logmel(clean)
        valid_frames = min(clean_mel.shape[-1], math.ceil(len(clean) / whisper.audio.HOP_LENGTH))
        clean_valid = clean_mel[:, :valid_frames]

        for condition in CONDITIONS:
            processed = read_mono_16k(audio_dir / f"{utterance_id}_{condition}.wav")
            if len(processed) != len(clean):
                raise ValueError(f"Length mismatch: {utterance_id} {condition}")
            processed_valid = official_logmel(processed)[:, :valid_frames]
            difference = processed_valid - clean_valid
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "condition": condition,
                    "valid_logmel_frames": valid_frames,
                    "logmel_mse": float(torch.mean(difference.square()).item()),
                    "logmel_mae": float(torch.mean(torch.abs(difference)).item()),
                }
            )

    per_utterance = pd.DataFrame(rows)
    summary = (
        per_utterance.groupby("condition", sort=False)
        .agg(
            n_utterances=("utterance_id", "count"),
            mean_logmel_mse=("logmel_mse", "mean"),
            std_logmel_mse=("logmel_mse", "std"),
            mean_logmel_mae=("logmel_mae", "mean"),
        )
        .reset_index()
    )

    per_utterance.to_csv(analysis_dir / "pilot_logmel_per_utterance.csv", index=False)
    summary.to_csv(analysis_dir / "pilot_logmel_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
