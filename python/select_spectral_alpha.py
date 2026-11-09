"""Choose spectral subtraction alpha using only the development set."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_alpha(method: str) -> float:
    match = re.fullmatch(r"spectral_alpha_(\d+)p(\d+)", method)
    if not match:
        raise ValueError(method)
    return float(f"{match.group(1)}.{match.group(2)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dev_root = root / "results" / "formal" / "development"
    wer = pd.read_csv(dev_root / "asr" / "wer_summary_overall.csv")
    metrics = pd.read_csv(dev_root / "metrics_per_file.csv")

    candidates = wer[wer.method.str.startswith("spectral_alpha_")].copy()
    candidates["alpha"] = candidates.method.map(parse_alpha)
    mean_metrics = (
        metrics[metrics.method.str.startswith("spectral_alpha_")]
        .groupby("method", as_index=False)
        .agg(
            mean_stoi=("stoiScore", "mean"),
            mean_output_snr_db=("outputSNRdB", "mean"),
        )
    )
    candidates = candidates.merge(mean_metrics, on="method", how="left")
    candidates["distance_from_original_alpha"] = (candidates.alpha - 1.0).abs()
    candidates = candidates.sort_values(
        ["wer_percent", "mean_stoi", "distance_from_original_alpha", "alpha"],
        ascending=[True, False, True, True],
    )
    selected = candidates.iloc[0]
    output = {
        "selected_alpha": float(selected.alpha),
        "selection_split": "development",
        "primary_rule": "lowest corpus-level WER across all development noise/SNR conditions",
        "tie_break_1": "highest mean STOI",
        "tie_break_2": "smallest absolute distance from original alpha=1.0",
        "candidate_table": candidates.to_dict("records"),
    }
    (dev_root / "selected_spectral_alpha.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    candidates.to_csv(dev_root / "spectral_alpha_selection.csv", index=False)
    print(candidates.to_string(index=False))
    print(f"Selected alpha: {selected.alpha:.2f}")


if __name__ == "__main__":
    main()
