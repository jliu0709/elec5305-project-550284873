"""Compare Whisper log-Mel features with the clean reference."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisper


def read_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite mono 16 kHz audio: {path}")
    return audio


def mel(audio: np.ndarray) -> torch.Tensor:
    return whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("split", choices=["development", "test"])
    parser.add_argument("--include-modified", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    result_root = root / "results" / "formal" / args.split
    mixtures = pd.read_csv(result_root / "mixture_manifest.csv")
    deep = pd.read_csv(result_root / "deepfilter_manifest.csv")
    deep_map = {
        (row.utterance_id, row.noise_type, int(row.target_snr_db)): row.deepfilter_relative_path
        for row in deep.itertuples()
    }
    modified_map: dict[tuple[str, str, int], str] = {}
    if args.include_modified:
        modified = pd.read_csv(result_root / "modified_spectral_manifest.csv")
        modified_map = {
            (row.utterance_id, row.noise_type, int(row.target_snr_db)): row.output_relative_path
            for row in modified.itertuples()
        }

    rows: list[dict[str, object]] = []
    for index, row in enumerate(mixtures.itertuples(), 1):
        key = (row.utteranceId, row.noiseType, int(row.targetSNRdB))
        clean = read_audio(root / row.cleanRelativePath) * float(row.commonScale)
        clean_mel = mel(clean)
        valid_frames = min(clean_mel.shape[-1], int(np.ceil(len(clean) / whisper.audio.HOP_LENGTH)))
        reference = clean_mel[:, :valid_frames]
        paths = {
            "noisy": row.noisyRelativePath,
            "spectral_subtraction": row.spectralRelativePath,
            "wiener": row.wienerRelativePath,
            "deepfilternet3": deep_map[key],
        }
        if args.include_modified:
            paths["spectral_subtraction_modified"] = modified_map[key]
        for method, relative_path in paths.items():
            processed = read_audio(root / relative_path)
            if len(processed) != len(clean):
                raise ValueError(f"Length mismatch: {key} {method}")
            difference = mel(processed)[:, :valid_frames] - reference
            rows.append(
                {
                    "utterance_id": row.utteranceId,
                    "noise_type": row.noiseType,
                    "target_snr_db": int(row.targetSNRdB),
                    "method": method,
                    "valid_logmel_frames": valid_frames,
                    "logmel_mse": float(torch.mean(difference.square()).item()),
                    "logmel_mae": float(torch.mean(torch.abs(difference)).item()),
                    "audio_relative_path": relative_path,
                }
            )
        if index % 25 == 0 or index == len(mixtures):
            print(f"[{index}/{len(mixtures)}] mixtures", flush=True)

    per_file = pd.DataFrame(rows)
    summary = (
        per_file.groupby(["noise_type", "target_snr_db", "method"], sort=False)
        .agg(
            n_files=("utterance_id", "count"),
            mean_logmel_mse=("logmel_mse", "mean"),
            std_logmel_mse=("logmel_mse", "std"),
            mean_logmel_mae=("logmel_mae", "mean"),
        )
        .reset_index()
    )
    per_file.to_csv(result_root / "logmel_per_file.csv", index=False)
    summary.to_csv(result_root / "logmel_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
