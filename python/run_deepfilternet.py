"""Run DeepFilterNet3 on the saved noisy audio."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import soundfile as sf
from scipy.signal import resample_poly


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("split", choices=["development", "test"])
    parser.add_argument("--binary", type=Path, help="Optional DeepFilterNet v0.5.6 executable path")
    args = parser.parse_args()

    root = args.project_root.resolve()
    default_name = (
        "deep-filter-0.5.6-x86_64-pc-windows-msvc.exe"
        if platform.system() == "Windows"
        else "deep-filter-0.5.6-x86_64-unknown-linux-gnu"
    )
    binary = args.binary.resolve() if args.binary else root / "tools" / default_name
    if not binary.is_file():
        raise FileNotFoundError(binary)

    result_root = root / "results" / "formal" / args.split
    manifest = pd.read_csv(result_root / "mixture_manifest.csv")
    manifest = manifest.rename(
        columns={
            "utteranceId": "utterance_id",
            "noiseType": "noise_type",
            "targetSNRdB": "target_snr_db",
            "noisyRelativePath": "noisy_relative_path",
        }
    )
    processing_root = result_root / "deepfilter_processing"
    processing_root.mkdir(parents=True, exist_ok=True)
    old_path = result_root / "deepfilter_manifest.csv"
    old = pd.read_csv(old_path) if old_path.is_file() else pd.DataFrame()
    old_by_key = {
        (row.utterance_id, row.noise_type, int(row.target_snr_db)): row
        for row in old.itertuples()
    }

    rows: list[dict[str, object]] = []
    for index, item in enumerate(manifest.itertuples(), 1):
        noisy_path = root / Path(item.noisy_relative_path)
        output_path = noisy_path.with_name(noisy_path.name.replace("_noisy.wav", "_deepfilter.wav"))
        key = (item.utterance_id, item.noise_type, int(item.target_snr_db))

        noisy, sample_rate = sf.read(noisy_path, dtype="float32")
        if sample_rate != 16000 or noisy.ndim != 1:
            raise ValueError(f"Expected mono 16 kHz audio: {noisy_path}")

        elapsed = float("nan")
        if not output_path.is_file():
            work_dir = processing_root / item.utterance_id / f"{item.noise_type}_{int(item.target_snr_db)}dB"
            input_dir = work_dir / "input"
            enhanced_dir = work_dir / "enhanced"
            input_dir.mkdir(parents=True, exist_ok=True)
            enhanced_dir.mkdir(parents=True, exist_ok=True)
            input_path = input_dir / "input_48k.wav"

            audio_48k = resample_poly(noisy, 3, 1).astype(np.float32, copy=False)
            original_length_48k = len(audio_48k)
            padded = np.concatenate([audio_48k, np.zeros(48000, dtype=np.float32)])
            sf.write(input_path, padded, 48000, subtype="FLOAT")

            started = time.perf_counter()
            process = subprocess.run(
                [str(binary), "-D", "-o", str(enhanced_dir), str(input_path)],
                capture_output=True,
                text=True,
            )
            elapsed = time.perf_counter() - started
            if process.returncode != 0:
                raise RuntimeError(f"DeepFilterNet failed for {key}:\n{process.stderr}")

            enhanced_path = enhanced_dir / input_path.name
            enhanced_48k, enhanced_rate = sf.read(enhanced_path, dtype="float32")
            if enhanced_rate != 48000 or enhanced_48k.ndim != 1:
                raise ValueError(f"Unexpected DeepFilterNet output: {enhanced_path}")
            enhanced_48k = enhanced_48k[:original_length_48k]
            enhanced_16k = resample_poly(enhanced_48k, 1, 3).astype(np.float32, copy=False)
            if len(enhanced_16k) != len(noisy):
                raise ValueError(f"DeepFilterNet length mismatch: {key}")
            if not np.isfinite(enhanced_16k).all():
                raise ValueError(f"DeepFilterNet produced invalid samples: {key}")
            sf.write(output_path, enhanced_16k, 16000, subtype="FLOAT")
            shutil.rmtree(work_dir)
        elif key in old_by_key:
            elapsed = float(old_by_key[key].processing_seconds)

        enhanced, enhanced_rate = sf.read(output_path, dtype="float32")
        if enhanced_rate != 16000 or len(enhanced) != len(noisy):
            raise ValueError(f"Saved DeepFilterNet output failed validation: {output_path}")
        rows.append(
            {
                "utterance_id": item.utterance_id,
                "project_split": args.split,
                "noise_type": item.noise_type,
                "target_snr_db": int(item.target_snr_db),
                "input_samples": len(noisy),
                "output_samples": len(enhanced),
                "output_peak": float(np.max(np.abs(enhanced))),
                "processing_seconds": elapsed,
                "real_time_factor": elapsed / (len(noisy) / 16000),
                "deepfilter_relative_path": output_path.relative_to(root).as_posix(),
            }
        )
        print(f"[{index}/{len(manifest)}] {key} complete", flush=True)

    pd.DataFrame(rows).to_csv(old_path, index=False)
    settings = {
        "release": "v0.5.6",
        "model": "DeepFilterNet3",
        "model_source": "embedded in official v0.5.6 binary",
        "binary": binary.name,
        "binary_sha256": sha256(binary),
        "resampling": "scipy.signal.resample_poly",
        "scipy_version": scipy.__version__,
        "sample_rates": "16000 -> 48000 -> 16000",
        "delay_compensation": True,
        "tail_padding_seconds": 1,
        "post_filter": False,
        "output_format": "32-bit float WAV",
    }
    (result_root / "deepfilter_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
