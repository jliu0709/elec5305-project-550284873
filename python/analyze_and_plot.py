"""Make the result tables, plots, and example cases."""

import argparse
import json
import shutil
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from whisper.normalizers import EnglishTextNormalizer


METHODS = [
    "noisy",
    "spectral_subtraction",
    "wiener",
    "deepfilternet3",
    "spectral_subtraction_modified",
]
LABELS = {
    "clean": "Clean",
    "noisy": "No enhancement",
    "spectral_subtraction": "Spectral subtraction (α=1.0)",
    "wiener": "Wiener",
    "deepfilternet3": "DeepFilterNet3",
    "spectral_subtraction_modified": "Spectral subtraction (α=1.5)",
}
COLORS = {
    "noisy": "#4D4D4D",
    "spectral_subtraction": "#0072B2",
    "wiener": "#009E73",
    "deepfilternet3": "#D55E00",
    "spectral_subtraction_modified": "#CC79A7",
}


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def pair_errors(reference: str, hypothesis: str, normalizer: EnglishTextNormalizer) -> dict[str, float | int]:
    result = jiwer.process_words(normalizer(reference), normalizer(hypothesis))
    words = result.hits + result.substitutions + result.deletions
    return {
        "errors": result.substitutions + result.deletions + result.insertions,
        "reference_words": words,
        "utterance_wer_percent": 100 * result.wer,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
    }


def make_case_spectrogram(case: pd.Series, transcripts: pd.DataFrame, root: Path, output: Path) -> None:
    keys = ["utterance_id", "noise_type", "target_snr_db"]
    subset = transcripts.copy()
    for key in keys:
        subset = subset[subset[key] == case[key]]
    noisy = subset[subset.method == "noisy"].iloc[0]
    enhanced = subset[subset.method == case.method].iloc[0]
    mixture = pd.read_csv(root / "results" / "formal" / "test" / "mixture_manifest.csv")
    match = mixture[
        (mixture.utteranceId == case.utterance_id)
        & (mixture.noiseType == case.noise_type)
        & (mixture.targetSNRdB == case.target_snr_db)
    ].iloc[0]
    paths = [
        ("Clean", root / match.cleanRelativePath),
        ("No enhancement", root / noisy.audio_relative_path),
        (LABELS[case.method], root / enhanced.audio_relative_path),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, sharey=True)
    for axis, (title, path) in zip(axes, paths):
        audio, fs = sf.read(path, dtype="float32")
        frequencies, times, spectrum = signal.spectrogram(audio, fs, nperseg=512, noverlap=384)
        use = frequencies <= 8000
        axis.pcolormesh(times, frequencies[use] / 1000, 10 * np.log10(spectrum[use] + 1e-10), shading="auto", cmap="magma", vmin=-100, vmax=-25)
        axis.set_ylabel("kHz")
        axis.set_title(title, loc="left", fontsize=10)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{case.case_type.title()} case: {case.utterance_id}, {case.noise_type}, {int(case.target_snr_db)} dB")
    fig.tight_layout()
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = root / "results" / "formal" / "test"
    figures = result / "figures"
    examples = result / "examples"
    figures.mkdir(parents=True, exist_ok=True)
    examples.mkdir(parents=True, exist_ok=True)

    transcripts = pd.read_csv(result / "asr" / "transcripts.csv")
    if len(transcripts) != 2300 or transcripts.duplicated(["utterance_id", "noise_type", "target_snr_db", "method"]).any():
        raise ValueError("Expected exactly 2,300 unique completed test recognition rows.")
    condition = pd.read_csv(result / "asr" / "wer_summary_by_condition.csv")
    overall = pd.read_csv(result / "asr" / "wer_summary_overall.csv")
    metrics = pd.read_csv(result / "metrics_summary.csv")
    metrics_per_file = pd.read_csv(result / "metrics_per_file.csv")
    logmel = pd.read_csv(result / "logmel_summary.csv")
    condition_metrics = condition.merge(metrics, on=["noise_type", "target_snr_db", "method"], how="left").merge(
        logmel, on=["noise_type", "target_snr_db", "method"], how="left"
    )
    condition_metrics.to_csv(result / "combined_condition_summary.csv", index=False)

    clean_wer = float(condition.loc[condition.method == "clean", "wer_percent"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)
    for axis, noise in zip(axes, ["white", "office", "traffic"]):
        subset = condition[condition.noise_type == noise]
        for method in METHODS:
            rows = subset[subset.method == method].sort_values("target_snr_db")
            axis.plot(rows.target_snr_db, rows.wer_percent, marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
        axis.axhline(clean_wer, color="#777777", linestyle="--", linewidth=1.3, label=f"Clean ({clean_wer:.1f}%)")
        axis.set_title(noise.title())
        axis.set_xlabel("Input SNR (dB)")
        axis.set_xticks([0, 5, 10])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Corpus WER (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.suptitle("Whisper base.en WER across noise types and input SNRs")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    save_figure(fig, figures / "01_wer_vs_snr")

    enhanced = [m for m in METHODS if m != "noisy"]
    pivot = condition[condition.method.isin(enhanced)].pivot(index=["noise_type", "target_snr_db"], columns="method", values="delta_wer_pp")
    pivot = pivot.reindex(pd.MultiIndex.from_product([["white", "office", "traffic"], [0, 5, 10]], names=pivot.index.names))
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.4), sharey=True)
    for axis, method in zip(axes, enhanced):
        image = pivot[method].unstack().loc[["white", "office", "traffic"], [0, 5, 10]]
        mesh = axis.imshow(image, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
        axis.set_title(LABELS[method], fontsize=9)
        axis.set_xticks(range(3), ["0", "5", "10"])
        axis.set_yticks(range(3), ["White", "Office", "Traffic"])
        axis.set_xlabel("Input SNR (dB)")
        for y in range(3):
            for x in range(3):
                axis.text(x, y, f"{image.iloc[y, x]:+.1f}", ha="center", va="center", fontsize=9)
    color_axis = fig.add_axes([0.925, 0.20, 0.015, 0.58])
    fig.colorbar(mesh, cax=color_axis, label="ΔWER (percentage points)")
    fig.suptitle("Change from matched unenhanced noisy speech (negative is better)")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.15, top=0.84, wspace=0.28)
    save_figure(fig, figures / "02_delta_wer_heatmap")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for column, noise in enumerate(["white", "office", "traffic"]):
        subset = metrics[metrics.noise_type == noise]
        for method in METHODS:
            rows = subset[subset.method == method].sort_values("target_snr_db")
            axes[0, column].plot(rows.target_snr_db, rows.mean_stoi, marker="o", color=COLORS[method], label=LABELS[method])
            axes[1, column].plot(rows.target_snr_db, rows.mean_output_snr_db, marker="o", color=COLORS[method])
        axes[0, column].set_title(noise.title())
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xlabel("Input SNR (dB)")
        axes[1, column].set_xticks([0, 5, 10])
    axes[0, 0].set_ylabel("Mean STOI")
    axes[1, 0].set_ylabel("Mean output SNR (dB)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle("Signal metrics by method and condition")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    save_figure(fig, figures / "03_stoi_and_output_snr")

    alpha = condition[condition.method.isin(["spectral_subtraction", "spectral_subtraction_modified"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, noise in zip(axes, ["white", "office", "traffic"]):
        subset = alpha[alpha.noise_type == noise]
        for method in ["spectral_subtraction", "spectral_subtraction_modified"]:
            rows = subset[subset.method == method].sort_values("target_snr_db")
            axis.plot(rows.target_snr_db, rows.wer_percent, marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
        axis.set_title(noise.title())
        axis.set_xlabel("Input SNR (dB)")
        axis.set_xticks([0, 5, 10])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Corpus WER (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.suptitle("Test-set evaluation of the development-selected spectral-subtraction modification")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, figures / "04_spectral_modification")

    scatter = condition_metrics[condition_metrics.method.isin(METHODS)].copy()
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    for method in METHODS:
        rows = scatter[scatter.method == method]
        axis.scatter(rows.mean_logmel_mse, rows.wer_percent, s=55, alpha=0.85, color=COLORS[method], label=LABELS[method])
    correlation = scatter[["mean_logmel_mse", "wer_percent"]].corr().iloc[0, 1]
    axis.set_xlabel("Mean official Whisper log-Mel MSE to scaled clean")
    axis.set_ylabel("Corpus WER (%)")
    axis.set_title(f"Feature distance and recognition error by condition (Pearson r={correlation:.2f})")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, figures / "05_logmel_vs_wer")

    timing = (
        metrics_per_file[metrics_per_file.method.isin(METHODS)]
        .groupby("method")
        .realTimeFactor.agg(median_rtf="median", p95_rtf=lambda values: values.quantile(0.95), max_rtf="max")
        .reindex(METHODS)
    )
    timing.to_csv(result / "enhancement_timing_summary.csv")
    asr_rtf = transcripts.groupby("method").asr_rtf.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].bar([LABELS[m] for m in METHODS], timing.median_rtf, color=[COLORS[m] for m in METHODS])
    axes[0].set_ylabel("Median enhancement RTF")
    axes[0].set_title("Front-end processing\n(median; raw outliers retained in CSV)", fontsize=10)
    axes[1].bar([LABELS[m] for m in METHODS], [asr_rtf[m] for m in METHODS], color=[COLORS[m] for m in METHODS])
    axes[1].set_ylabel("Whisper RTF")
    axes[1].set_title("Recognition")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=58)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Real-time factors on the experiment machine", y=0.96)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.40, wspace=0.35)
    save_figure(fig, figures / "06_runtime_rtf")

    normalizer = EnglishTextNormalizer()
    scored = []
    for row in transcripts[transcripts.method != "clean"].itertuples():
        scored.append({**row._asdict(), **pair_errors(row.reference_text, row.hypothesis_text, normalizer)})
    scored_df = pd.DataFrame(scored)
    utterance_totals = (
        scored_df.groupby(["utterance_id", "method"])[["errors", "reference_words"]]
        .sum()
        .reset_index()
    )
    total_error = utterance_totals.pivot(index="utterance_id", columns="method", values="errors")
    total_words = utterance_totals[utterance_totals.method == "noisy"].set_index("utterance_id").reference_words
    rng = np.random.default_rng(5305)
    ids = total_error.index.to_numpy()
    bootstrap_rows = []
    for method in enhanced:
        estimates = np.empty(10000)
        for replicate in range(len(estimates)):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            numerator = (total_error.loc[sampled, method].to_numpy() - total_error.loc[sampled, "noisy"].to_numpy()).sum()
            denominator = total_words.loc[sampled].to_numpy().sum()
            estimates[replicate] = 100 * numerator / denominator
        observed = 100 * (total_error[method].sum() - total_error["noisy"].sum()) / total_words.sum()
        bootstrap_rows.append(
            {
                "method": method,
                "observed_delta_wer_pp": observed,
                "cluster_bootstrap_ci95_low_pp": np.quantile(estimates, 0.025),
                "cluster_bootstrap_ci95_high_pp": np.quantile(estimates, 0.975),
                "bootstrap_replicates": len(estimates),
                "cluster_unit": "utterance",
            }
        )
    pd.DataFrame(bootstrap_rows).to_csv(result / "delta_wer_cluster_bootstrap.csv", index=False)
    noisy = scored_df[scored_df.method == "noisy"][
        ["utterance_id", "noise_type", "target_snr_db", "errors", "utterance_wer_percent", "hypothesis_text", "audio_relative_path"]
    ].rename(columns={c: f"noisy_{c}" for c in ["errors", "utterance_wer_percent", "hypothesis_text", "audio_relative_path"]})
    candidates = scored_df[scored_df.method.isin(enhanced)].merge(noisy, on=["utterance_id", "noise_type", "target_snr_db"])
    candidates["delta_errors"] = candidates.errors - candidates.noisy_errors
    candidates["delta_utterance_wer_pp"] = candidates.utterance_wer_percent - candidates.noisy_utterance_wer_percent
    success = candidates.sort_values(["delta_errors", "delta_utterance_wer_pp"]).iloc[0].copy()
    failure = candidates.sort_values(["delta_errors", "delta_utterance_wer_pp"], ascending=False).iloc[0].copy()
    cases = []
    for case_type, row in [("success", success), ("failure", failure)]:
        row["case_type"] = case_type
        cases.append(row)
        method_token = row.method.replace("spectral_subtraction_modified", "spectral_modified")
        noisy_source = root / row.noisy_audio_relative_path
        enhanced_source = root / row.audio_relative_path
        shutil.copy2(noisy_source, examples / f"{case_type}_{row.utterance_id}_noisy.wav")
        shutil.copy2(enhanced_source, examples / f"{case_type}_{row.utterance_id}_{method_token}.wav")
    case_table = pd.DataFrame(cases)
    keep = [
        "case_type", "utterance_id", "noise_type", "target_snr_db", "method", "reference_text",
        "noisy_hypothesis_text", "hypothesis_text", "noisy_errors", "errors", "delta_errors",
        "substitutions", "deletions", "insertions",
        "noisy_utterance_wer_percent", "utterance_wer_percent", "delta_utterance_wer_pp",
        "noisy_audio_relative_path", "audio_relative_path",
    ]
    case_table[keep].to_csv(examples / "selected_cases.csv", index=False, encoding="utf-8-sig")

    # Check whether the worst failure alone explains the method-level result.
    compared_methods = ["noisy", failure.method]
    comparison = scored_df[scored_df.method.isin(compared_methods)].copy()
    matched_failure = (
        (comparison.utterance_id == failure.utterance_id)
        & (comparison.noise_type == failure.noise_type)
        & (comparison.target_snr_db == failure.target_snr_db)
    )
    sensitivity_rows = []
    for label, frame in [("all_rows", comparison), ("matched_failure_removed", comparison[~matched_failure])]:
        for method, group in frame.groupby("method"):
            errors = int(group.errors.sum())
            reference_words = int(group.reference_words.sum())
            sensitivity_rows.append(
                {
                    "analysis": label,
                    "method": method,
                    "errors": errors,
                    "reference_words": reference_words,
                    "wer_percent": 100 * errors / reference_words,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    noisy_wer = sensitivity[sensitivity.method == "noisy"].set_index("analysis").wer_percent
    sensitivity["delta_wer_pp"] = [
        row.wer_percent - noisy_wer[row.analysis] for row in sensitivity.itertuples()
    ]
    sensitivity.to_csv(examples / "failure_case_sensitivity.csv", index=False)

    for row in case_table.itertuples():
        make_case_spectrogram(pd.Series(row._asdict()), transcripts, root, figures / f"07_{row.case_type}_spectrogram")

    findings = {
        "test_transcription_rows": len(transcripts),
        "clean_wer_percent": clean_wer,
        "overall_methods": overall.to_dict("records"),
        "feature_wer_condition_correlation": float(correlation),
        "selected_success": case_table.iloc[0][keep[:-2]].to_dict(),
        "selected_failure": case_table.iloc[1][keep[:-2]].to_dict(),
        "failure_case_sensitivity": sensitivity.to_dict("records"),
    }
    with (result / "analysis_findings.json").open("w", encoding="utf-8") as handle:
        json.dump(findings, handle, indent=2, ensure_ascii=False, default=str)
    print(overall.to_string(index=False))
    print(case_table[keep[:-2]].to_string(index=False))


if __name__ == "__main__":
    main()
