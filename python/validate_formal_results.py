"""Check the saved development and test results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()

    development = pd.read_csv(root / "data" / "splits" / "development_manifest.csv")
    test = pd.read_csv(root / "data" / "splits" / "test_manifest.csv")
    assert len(development) == 5 and len(test) == 50
    assert set(development.speaker_id).isdisjoint(set(test.speaker_id))

    report: dict[str, object] = {
        "development_utterances": len(development),
        "test_utterances": len(test),
        "development_speakers": development.speaker_id.nunique(),
        "test_speakers": test.speaker_id.nunique(),
        "speaker_overlap": 0,
    }
    for split, expected_mixtures, expected_transcripts, expected_metrics in [
        ("development", 45, 410, 405),
        ("test", 450, 2300, 2250),
    ]:
        result = root / "results" / "formal" / split
        mixtures = pd.read_csv(result / "mixture_manifest.csv")
        deep = pd.read_csv(result / "deepfilter_manifest.csv")
        transcripts = pd.read_csv(result / "asr" / "transcripts.csv")
        metrics = pd.read_csv(result / "metrics_per_file.csv")
        assert len(mixtures) == expected_mixtures and len(deep) == expected_mixtures
        assert len(transcripts) == expected_transcripts and len(metrics) == expected_metrics
        assert not transcripts.duplicated(["utterance_id", "noise_type", "target_snr_db", "method"]).any()
        assert not metrics.duplicated(["utteranceId", "noiseType", "targetSNRdB", "method"]).any()
        assert np.max(np.abs(mixtures.measuredInputSNRdB - mixtures.targetSNRdB)) < 0.01
        assert mixtures.reconstructionError.max() < 1e-9
        assert metrics.estimatedLagSamples.abs().max() == 0
        assert metrics.peakAmplitude.max() < 1
        for relative in transcripts.audio_relative_path:
            path = root / relative
            assert path.is_file(), path
            info = sf.info(path)
            assert info.samplerate == 16000 and info.channels == 1
        report[split] = {
            "mixtures": len(mixtures),
            "transcripts": len(transcripts),
            "waveform_metric_rows": len(metrics),
            "max_abs_target_snr_error_db": float(np.max(np.abs(mixtures.measuredInputSNRdB - mixtures.targetSNRdB))),
            "max_reconstruction_error": float(mixtures.reconstructionError.max()),
            "max_abs_alignment_lag_samples": int(metrics.estimatedLagSamples.abs().max()),
            "max_peak_amplitude": float(metrics.peakAmplitude.max()),
        }
    output = root / "results" / "formal" / "validation_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
