"""Summarise benchmark results and create experiment figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("results_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(f"No results_*.jsonl files found in {results_dir}")
    frame = pd.DataFrame(rows)
    frame["position_label"] = pd.Categorical(
        frame["position_label"],
        categories=["beginning", "early", "middle", "late", "end"],
        ordered=True,
    )
    return frame


def write_analysis(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (frame.groupby(["model", "size", "similarity", "position_label"],
                             observed=True)["correct"]
               .agg(["mean", "count", "sum"])
               .reset_index()
               .rename(columns={"mean": "accuracy", "sum": "correct_trials"}))
    summary.to_csv(output_dir / "accuracy_summary.csv", index=False)
    errors = (frame.assign(is_error=frame["correct"].eq(0))
              .groupby(["model", "size", "similarity", "error_type"],
                       observed=True)["is_error"].mean()
              .reset_index(name="rate"))
    errors.to_csv(output_dir / "error_type_rates.csv", index=False)

    sns.set_theme(style="whitegrid")
    for model, subset in frame.groupby("model"):
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=subset, x="position_label", y="correct", hue="similarity",
                     estimator="mean", errorbar=("ci", 95), marker="o")
        plt.ylim(0, 1.05)
        plt.ylabel("Accuracy")
        plt.xlabel("Target position")
        plt.title(f"Accuracy by target position: {model}")
        plt.tight_layout()
        plt.savefig(output_dir / f"accuracy_by_position_{model.replace(':', '_')}.png", dpi=180)
        plt.close()

    heatmap = (frame.groupby(["size", "similarity", "position_label"], observed=True)["correct"]
               .mean().reset_index()
               .pivot_table(index=["size", "similarity"], columns="position_label",
                            values="correct", observed=True))
    plt.figure(figsize=(9, 5))
    sns.heatmap(heatmap, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis")
    plt.title("Accuracy by size, similarity, and target position")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_heatmap.png", dpi=180)
    plt.close()

    # Keep the inferential result reproducible without making analysis fail if a
    # local statsmodels build lacks a mixed-effects implementation.
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        model = smf.glm("correct ~ C(position_label) * C(similarity) * C(size)",
                        data=frame, family=sm.families.Binomial()).fit()
        (output_dir / "glm_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    except (ImportError, ValueError, KeyError) as exc:
        (output_dir / "glm_summary.txt").write_text(
            f"GLM unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    frame = load_results(args.results_dir)
    write_analysis(frame, args.output_dir)
    print(f"Analysed {len(frame)} trials; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
