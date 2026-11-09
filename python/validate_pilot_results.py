"""Check the pilot transcripts and calculate corpus WER again."""

import argparse
import csv
import json
from pathlib import Path

import jiwer
import pandas as pd
from whisper.normalizers import EnglishTextNormalizer


TRANSCRIPT_FILES = {
    "clean": "clean_transcripts.csv",
    "noisy": "noisy_transcripts.csv",
    "spectral_subtraction": "spectral_subtraction_transcripts.csv",
    "wiener": "wiener_transcripts.csv",
    "deepfilternet3": "deepfilter_transcripts.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    asr_dir = root / "results" / "asr"
    analysis_dir = root / "results" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with (root / "utterances.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    expected_ids = [row["utterance_id"] for row in manifest]
    expected_refs = [row["reference_text"] for row in manifest]
    normalizer = EnglishTextNormalizer()
    normalized_refs = [normalizer(text) for text in expected_refs]

    summary_rows = []
    for condition, filename in TRANSCRIPT_FILES.items():
        with (asr_dir / filename).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row["utterance_id"] for row in rows]
        if ids != expected_ids:
            raise ValueError(f"Utterance order mismatch in {filename}")
        stored_refs = [row["reference_text"] for row in rows]
        if stored_refs != expected_refs:
            raise ValueError(f"Reference mismatch in {filename}")

        normalized_hypotheses = [normalizer(row["hypothesis_text"]) for row in rows]
        score = jiwer.process_words(normalized_refs, normalized_hypotheses)
        summary_rows.append(
            {
                "condition": condition,
                "wer_percent": score.wer * 100,
                "substitutions": score.substitutions,
                "deletions": score.deletions,
                "insertions": score.insertions,
                "reference_words": score.hits + score.substitutions + score.deletions,
            }
        )

    noisy_wer = next(row["wer_percent"] for row in summary_rows if row["condition"] == "noisy")
    for row in summary_rows:
        row["delta_wer_pp"] = row["wer_percent"] - noisy_wer
        row["relative_wer_change_percent"] = (
            100 * (row["wer_percent"] - noisy_wer) / noisy_wer if noisy_wer else float("nan")
        )

    validated = pd.DataFrame(summary_rows)
    validated.to_csv(analysis_dir / "pilot_wer_validated.csv", index=False)

    settings = json.loads((asr_dir / "recognition_settings.json").read_text(encoding="utf-8"))
    if settings.get("model") != "base.en":
        raise ValueError("Pilot was not run with Whisper base.en")
    print(validated.to_string(index=False))
    print("Recognition settings validated: base.en with fixed saved decode settings.")


if __name__ == "__main__":
    main()
