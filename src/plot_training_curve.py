from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import TRAINING_CURVE_CSV, TRAINING_CURVE_PNG, TRAINING_METRICS, ensure_dirs


def moving_average(values: list[float], window: int = 25) -> list[float]:
    output = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output.append(sum(values[start : index + 1]) / (index - start + 1))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retriever training convergence.")
    parser.add_argument("--metrics", type=Path, default=TRAINING_METRICS)
    parser.add_argument("--csv", type=Path, default=TRAINING_CURVE_CSV)
    parser.add_argument("--png", type=Path, default=TRAINING_CURVE_PNG)
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()

    ensure_dirs()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = metrics.get("step_history", [])
    if not rows:
        rows = [
            {"step": row["epoch"], "epoch": row["epoch"], "loss": row["loss"]}
            for row in metrics.get("history", [])
        ]
    losses = [float(row["loss"]) for row in rows]
    smooth = moving_average(losses, args.window)

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "loss", "smooth_loss"])
        writer.writeheader()
        for row, smooth_loss in zip(rows, smooth):
            writer.writerow(
                {
                    "step": row["step"],
                    "epoch": row["epoch"],
                    "loss": row["loss"],
                    "smooth_loss": smooth_loss,
                }
            )

    try:
        import matplotlib.pyplot as plt

        steps = [int(row["step"]) for row in rows]
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, losses, alpha=0.35, label="batch loss")
        plt.plot(steps, smooth, linewidth=2, label=f"moving average ({args.window})")
        plt.xlabel("Training step")
        plt.ylabel("InfoNCE loss")
        plt.title("Retriever Fine-tuning Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.png, dpi=200)
        print(f"Saved PNG: {args.png}")
    except Exception as error:
        print(f"WARNING: could not render PNG: {error}")
    print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
