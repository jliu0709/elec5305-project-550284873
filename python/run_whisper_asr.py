"""Run Whisper base.en and save progress after each file."""

import argparse
import csv
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisper
from whisper.normalizers import EnglishTextNormalizer


BASE_METHOD_COLUMNS = {
    "noisy": "noisyRelativePath",
    "spectral_subtraction": "spectralRelativePath",
    "wiener": "wienerRelativePath",
}
ALPHAS = [0.75, 1.00, 1.25, 1.50, 2.00]


def alpha_label(alpha: float) -> str:
    return f"spectral_alpha_{alpha:.2f}".replace(".", "p")


def alpha_filename(stem: str, alpha: float) -> str:
    token = f"{alpha:.2f}".replace(".", "p")
    return f"{stem}_spectral_alpha_{token}.wav"


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid Whisper input: {path}")
    return audio


def resume_snr(value: object) -> int | None:
    """Use one value for the missing SNR in clean-speech rows."""
    return None if pd.isna(value) else int(value)


def build_tasks(root: Path, split: str, include_alpha_sweep: bool, modified_alpha: float | None):
    split_manifest = pd.read_csv(root / "data" / "splits" / f"{split}_manifest.csv")
    mixture = pd.read_csv(root / "results" / "formal" / split / "mixture_manifest.csv")
    deepfilter = pd.read_csv(root / "results" / "formal" / split / "deepfilter_manifest.csv")
    deepfilter_map = {
        (row.utterance_id, row.noise_type, int(row.target_snr_db)): row.deepfilter_relative_path
        for row in deepfilter.itertuples()
    }
    reference_map = dict(zip(split_manifest.utterance_id, split_manifest.reference_text))
    tasks = []
    for row in split_manifest.itertuples():
        tasks.append(
            {
                "utterance_id": row.utterance_id,
                "noise_type": "clean",
                "target_snr_db": math.nan,
                "method": "clean",
                "path": root / row.clean_wav,
                "reference_text": row.reference_text,
            }
        )

    for row in mixture.itertuples():
        key = (row.utteranceId, row.noiseType, int(row.targetSNRdB))
        for method, column in BASE_METHOD_COLUMNS.items():
            tasks.append(
                {
                    "utterance_id": row.utteranceId,
                    "noise_type": row.noiseType,
                    "target_snr_db": int(row.targetSNRdB),
                    "method": method,
                    "path": root / getattr(row, column),
                    "reference_text": reference_map[row.utteranceId],
                }
            )
        tasks.append(
            {
                "utterance_id": row.utteranceId,
                "noise_type": row.noiseType,
                "target_snr_db": int(row.targetSNRdB),
                "method": "deepfilternet3",
                "path": root / deepfilter_map[key],
                "reference_text": reference_map[row.utteranceId],
            }
        )
        stem = f"{row.noiseType}_{int(row.targetSNRdB)}dB"
        utterance_dir = root / "results" / "formal" / split / "audio" / row.utteranceId
        if include_alpha_sweep:
            for alpha in ALPHAS:
                tasks.append(
                    {
                        "utterance_id": row.utteranceId,
                        "noise_type": row.noiseType,
                        "target_snr_db": int(row.targetSNRdB),
                        "method": alpha_label(alpha),
                        "path": utterance_dir / alpha_filename(stem, alpha),
                        "reference_text": reference_map[row.utteranceId],
                    }
                )
        if modified_alpha is not None:
            tasks.append(
                {
                    "utterance_id": row.utteranceId,
                    "noise_type": row.noiseType,
                    "target_snr_db": int(row.targetSNRdB),
                    "method": "spectral_subtraction_modified",
                    "path": utterance_dir / alpha_filename(stem, modified_alpha),
                    "reference_text": reference_map[row.utteranceId],
                }
            )
    return tasks


def summarize(transcripts: pd.DataFrame, result_root: Path) -> None:
    normalizer = EnglishTextNormalizer()

    def score_group(group: pd.DataFrame) -> dict[str, float | int]:
        refs = [normalizer(value) for value in group.reference_text]
        hyps = [normalizer(value) for value in group.hypothesis_text]
        score = jiwer.process_words(refs, hyps)
        return {
            "wer_percent": score.wer * 100,
            "substitutions": score.substitutions,
            "deletions": score.deletions,
            "insertions": score.insertions,
            "reference_words": score.hits + score.substitutions + score.deletions,
            "mean_asr_seconds": group.asr_seconds.mean(),
            "mean_asr_rtf": group.asr_rtf.mean(),
        }

    condition_rows = []
    for keys, group in transcripts.groupby(["noise_type", "target_snr_db", "method"], dropna=False):
        noise, snr, method = keys
        condition_rows.append({"noise_type": noise, "target_snr_db": snr, "method": method, **score_group(group)})
    by_condition = pd.DataFrame(condition_rows)
    noisy_lookup = {
        (row.noise_type, row.target_snr_db): row.wer_percent
        for row in by_condition.itertuples()
        if row.method == "noisy"
    }
    by_condition["delta_wer_pp"] = [
        (row.wer_percent - noisy_lookup[(row.noise_type, row.target_snr_db)])
        if row.method != "clean" and (row.noise_type, row.target_snr_db) in noisy_lookup
        else math.nan
        for row in by_condition.itertuples()
    ]
    by_condition["relative_wer_change_percent"] = [
        100 * row.delta_wer_pp / noisy_lookup[(row.noise_type, row.target_snr_db)]
        if not math.isnan(row.delta_wer_pp) and noisy_lookup[(row.noise_type, row.target_snr_db)] != 0
        else math.nan
        for row in by_condition.itertuples()
    ]
    by_condition.to_csv(result_root / "wer_summary_by_condition.csv", index=False)

    overall_rows = []
    for method, group in transcripts[transcripts.method != "clean"].groupby("method", sort=False):
        overall_rows.append({"method": method, **score_group(group)})
    overall = pd.DataFrame(overall_rows)
    noisy_overall = float(overall.loc[overall.method == "noisy", "wer_percent"].iloc[0])
    overall["delta_wer_pp"] = overall.wer_percent - noisy_overall
    overall["relative_wer_change_percent"] = 100 * overall.delta_wer_pp / noisy_overall
    overall.to_csv(result_root / "wer_summary_overall.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("split", choices=["development", "test"])
    parser.add_argument("--include-alpha-sweep", action="store_true")
    parser.add_argument("--modified-alpha", type=float)
    args = parser.parse_args()

    root = args.project_root.resolve()
    result_root = root / "results" / "formal" / args.split / "asr"
    result_root.mkdir(parents=True, exist_ok=True)
    transcript_path = result_root / "transcripts.csv"
    existing = pd.read_csv(transcript_path) if transcript_path.is_file() else pd.DataFrame()
    done = {
        (row.utterance_id, row.noise_type, resume_snr(row.target_snr_db), row.method)
        for row in existing.itertuples()
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(5305)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(5305)
    model = whisper.load_model("base.en", device=device)
    model.eval()
    decode_settings = {
        "language": "en",
        "task": "transcribe",
        "temperature": 0.0,
        "beam_size": 5,
        "fp16": False,
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "verbose": None,
    }

    tasks = build_tasks(root, args.split, args.include_alpha_sweep, args.modified_alpha)
    rows = existing.to_dict("records") if not existing.empty else []
    for index, task in enumerate(tasks, 1):
        key = (
            task["utterance_id"],
            task["noise_type"],
            resume_snr(task["target_snr_db"]),
            task["method"],
        )
        if key in done:
            continue
        path = Path(task["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        audio = load_audio(path)
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.transcribe(audio, **decode_settings)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "utterance_id": task["utterance_id"],
                "project_split": args.split,
                "noise_type": task["noise_type"],
                "target_snr_db": task["target_snr_db"],
                "method": task["method"],
                "reference_text": task["reference_text"],
                "hypothesis_text": result["text"].strip(),
                "audio_seconds": len(audio) / 16000,
                "asr_seconds": elapsed,
                "asr_rtf": elapsed / (len(audio) / 16000),
                "audio_relative_path": path.relative_to(root).as_posix(),
            }
        )
        pd.DataFrame(rows).to_csv(transcript_path, index=False, encoding="utf-8-sig")
        print(f"[{index}/{len(tasks)}] {key} -> {result['text'].strip()}", flush=True)

    transcripts = pd.DataFrame(rows)
    summarize(transcripts, result_root)
    settings = {
        "model": "base.en",
        "device": device,
        "decode_settings": decode_settings,
        "normalizer": "Whisper EnglishTextNormalizer",
        "openai_whisper_version": version("openai-whisper"),
        "jiwer_version": version("jiwer"),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": 5305,
    }
    (result_root / "recognition_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
