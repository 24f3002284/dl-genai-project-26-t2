"""
Pulls the three REQUIRED WandB runs (model1-scratch, model2-bert,
model3-bilstm) from the shared project and builds a single comparison
table + bar chart on the common metrics (accuracy, macro-F1, MAP@3) --
this is what the "at least three runs must be compared" grading criterion
is asking for. Add "model3-deberta" to RUN_NAMES to include the
exploratory extra Transformer run in the same comparison.

Requires WANDB_API_KEY to be set (same auth as the training scripts) and
your WandB entity (username/team). Pass it explicitly if it's not your
default.

Usage:
    python src/compare_runs.py --entity YOUR_WANDB_ENTITY
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from config import WANDB_PROJECT

RUN_NAMES = ["model1-scratch", "model2-bert", "model3-bilstm"]
METRICS = ["val_acc", "val_f1", "val_map3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=None, help="WandB username/team (defaults to your logged-in entity)")
    p.add_argument("--output-dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    api = wandb.Api()
    project_path = f"{args.entity}/{WANDB_PROJECT}" if args.entity else WANDB_PROJECT
    runs = {r.name: r for r in api.runs(project_path)}

    rows = []
    for name in RUN_NAMES:
        if name not in runs:
            print(f"WARNING: run '{name}' not found in {project_path} -- skipping")
            continue
        summary = runs[name].summary
        row = {"run": name}
        for m in METRICS:
            row[m] = summary.get(m, None)
        row["best_val_f1"] = summary.get("best_val_f1", None)
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No matching runs found. Check --entity and that all three "
            "training scripts have been run with wandb online (not offline)."
        )

    df = pd.DataFrame(rows).set_index("run")
    csv_path = os.path.join(args.output_dir, "run_comparison.csv")
    df.to_csv(csv_path)
    print(df)
    print(f"\nSaved table to {csv_path}")

    # Bar chart of accuracy vs F1 vs MAP@3 across the three runs
    ax = df[METRICS].plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Score")
    ax.set_title("Validation metrics across the three models")
    ax.set_xticklabels(df.index, rotation=15)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "run_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved chart to {plot_path}")


if __name__ == "__main__":
    main()
