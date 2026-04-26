from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from biorag.io import read_json, write_json
from biorag.utils import ensure_dir


def _moving_average(values: list[float], window: int) -> list[float]:
    active_window = max(1, window)
    smoothed: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= active_window:
            running_sum -= values[index - active_window]
        denominator = min(index + 1, active_window)
        smoothed.append(running_sum / denominator)
    return smoothed


def _curve_rows(metrics: dict[str, Any], smoothing_window: int) -> list[dict[str, float | int]]:
    step_history = metrics.get("step_history") or []
    if step_history:
        rows = [
            {
                "step": int(row["step"]),
                "epoch": int(row.get("epoch", 0)),
                "batch": int(row.get("batch", 0)),
                "optimizer_step": int(row.get("optimizer_step", 0)),
                "loss": float(row["loss"]),
            }
            for row in step_history
        ]
    else:
        rows = [
            {
                "step": int(row["epoch"]),
                "epoch": int(row["epoch"]),
                "batch": 0,
                "optimizer_step": int(row.get("optimizer_steps", row["epoch"])),
                "loss": float(row["loss"]),
            }
            for row in metrics.get("history", [])
        ]
    smoothed = _moving_average([float(row["loss"]) for row in rows], smoothing_window)
    for row, value in zip(rows, smoothed):
        row["smoothed_loss"] = value
    return rows


def _write_curve_csv(path: Path, rows: list[dict[str, float | int]]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "epoch",
                "batch",
                "optimizer_step",
                "loss",
                "smoothed_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _render_curve_png(
    path: Path,
    rows: list[dict[str, float | int]],
    *,
    title: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: "
            "python -m pip install matplotlib"
        ) from error

    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    smoothed = [float(row["smoothed_loss"]) for row in rows]

    ensure_dir(path.parent)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    ax.plot(steps, losses, color="#7aa6c2", alpha=0.22, linewidth=0.9, label="Batch loss")
    ax.plot(steps, smoothed, color="#1f5f8b", linewidth=2.2, label="Smoothed loss")
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Contrastive loss")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.32)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_training_curve(
    training_metrics: str | Path,
    output_dir: str | Path,
    *,
    smoothing_window: int = 25,
    title: str = "Retriever Training Loss",
    render_image: bool = True,
) -> dict[str, Path]:
    metrics = read_json(training_metrics)
    rows = _curve_rows(metrics, smoothing_window)
    if not rows:
        raise ValueError("No training history was found in the metrics file.")
    root = ensure_dir(output_dir)
    csv_path = _write_curve_csv(root / "training_loss_curve.csv", rows)
    json_path = write_json(
        root / "training_loss_curve.json",
        {
            "source": str(training_metrics),
            "smoothing_window": smoothing_window,
            "points": len(rows),
            "min_loss": min(float(row["loss"]) for row in rows),
            "final_loss": float(rows[-1]["loss"]),
        },
    )
    artifacts = {"csv": csv_path, "json": json_path}
    if render_image:
        artifacts["png"] = _render_curve_png(
            root / "training_loss_curve.png",
            rows,
            title=title,
        )
    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render BioRAG training convergence curves.")
    parser.add_argument(
        "--training-metrics",
        required=True,
        help="Path to training_metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write curve artifacts into.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=25,
        help="Trailing moving-average window for the plotted curve.",
    )
    parser.add_argument(
        "--title",
        default="Retriever Training Loss",
        help="Figure title.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    write_training_curve(
        args.training_metrics,
        args.output_dir,
        smoothing_window=args.smoothing_window,
        title=args.title,
    )


if __name__ == "__main__":
    main()
