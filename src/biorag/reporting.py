from __future__ import annotations

import argparse
from pathlib import Path

from biorag.io import read_json, write_json, write_text
from biorag.utils import ensure_dir

METRIC_SPECS = [
    ("Yes/No", "Accuracy", "yesno_accuracy"),
    ("Yes/No", "Macro-F1", "yesno_macro_f1"),
    ("Factoid", "Strict Acc", "factoid_strict_accuracy"),
    ("List", "List Recall", "list_recall"),
    ("Overall", "BERTScore", "bert_score"),
    ("Overall", "Latency (s)", "average_latency_seconds"),
]


def _load_metrics(report_path: str | Path) -> dict[str, float | None]:
    payload = read_json(report_path)
    return payload.get("metrics", {})


def _format_value(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.4f}"


def _format_gain(metric_key: str, baseline: float | None, ours: float | None) -> str:
    if baseline is None or ours is None:
        return "null"
    if metric_key == "average_latency_seconds":
        delta = ours - baseline
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.2f}s"
    if baseline == 0:
        delta = ours - baseline
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta * 100:.1f}%"
    relative = ((ours - baseline) / baseline) * 100
    sign = "+" if relative >= 0 else ""
    return f"{sign}{relative:.1f}%"


def build_report_comparison(
    baseline_report: str | Path,
    ours_report: str | Path,
) -> tuple[list[dict[str, str | float | None]], str]:
    baseline_metrics = _load_metrics(baseline_report)
    ours_metrics = _load_metrics(ours_report)
    rows: list[dict[str, str | float | None]] = []
    markdown_lines = [
        "| Question Type | Metric | Baseline | Ours | Gain |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for question_type, metric_label, metric_key in METRIC_SPECS:
        baseline_value = baseline_metrics.get(metric_key)
        ours_value = ours_metrics.get(metric_key)
        gain = _format_gain(metric_key, baseline_value, ours_value)
        row = {
            "question_type": question_type,
            "metric": metric_label,
            "metric_key": metric_key,
            "baseline": baseline_value,
            "ours": ours_value,
            "gain": gain,
        }
        rows.append(row)
        markdown_lines.append(
            "| "
            f"{question_type} | {metric_label} | {_format_value(baseline_value)} | "
            f"{_format_value(ours_value)} | {gain} |"
        )
    return rows, "\n".join(markdown_lines) + "\n"


def write_report_comparison(
    baseline_report: str | Path,
    ours_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    rows, markdown = build_report_comparison(baseline_report, ours_report)
    root = ensure_dir(output_dir)
    markdown_path = write_text(root / "paper_comparison.md", markdown)
    json_path = write_json(root / "paper_comparison.json", {"rows": rows})
    return {"markdown": markdown_path, "json": json_path}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render paper-style BioRAG comparison tables.")
    parser.add_argument(
        "--baseline-report",
        required=True,
        help="Path to the baseline evaluation report JSON.",
    )
    parser.add_argument(
        "--ours-report",
        required=True,
        help="Path to the BioRAG evaluation report JSON.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the comparison artifacts into.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    write_report_comparison(args.baseline_report, args.ours_report, args.output_dir)


if __name__ == "__main__":
    main()
