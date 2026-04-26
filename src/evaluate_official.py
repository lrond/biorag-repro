from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config import (
    BASELINE_PREDICTIONS,
    EVAL_PROCESSED,
    FULL_PREDICTIONS,
    RESULTS_JSON,
    RESULTS_MD,
    ensure_dirs,
)
from data_utils import load_jsonl


def normalize(text: Any) -> str:
    text = str(text or "").lower().strip()
    text = text.replace("assistant\n", "")
    return re.sub(r"\s+", " ", text)


def normalize_match(text: Any) -> str:
    return re.sub(r"[^\w\s]", "", normalize(text)).strip()


def flatten_exact(raw: Any) -> list[list[str]]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [[raw]]
    if isinstance(raw, list):
        if not raw:
            return []
        if all(isinstance(item, str) for item in raw):
            return [[str(item)] for item in raw]
        groups = []
        for item in raw:
            if isinstance(item, list):
                values = [str(value) for value in item if str(value).strip()]
                if values:
                    groups.append(values)
            elif str(item).strip():
                groups.append([str(item)])
        return groups
    return [[str(raw)]]


def rouge_l(candidate: str, reference: str) -> float:
    cand = normalize(candidate).split()
    ref = normalize(reference).split()
    if not cand or not ref:
        return 0.0
    table = [[0] * (len(ref) + 1) for _ in range(len(cand) + 1)]
    for i, token_c in enumerate(cand, start=1):
        for j, token_r in enumerate(ref, start=1):
            if token_c == token_r:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def yesno_label(text: str) -> str:
    norm = normalize(text)
    if norm.startswith("yes"):
        return "yes"
    if norm.startswith("no"):
        return "no"
    has_yes = re.search(r"\byes\b", norm)
    has_no = re.search(r"\bno\b", norm)
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "unknown"


def split_prediction_items(answer: str) -> list[str]:
    parts = [
        part.strip(" -\t\n\r")
        for part in re.split(r"[\n;,]", str(answer or ""))
        if part.strip(" -\t\n\r")
    ]
    return parts or [str(answer or "").strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores)


def group_hit(candidate: str, group: set[str]) -> bool:
    cand = normalize_match(candidate)
    return any(gold and (gold in cand or cand in gold) for gold in group)


def score_predictions(prediction_path: Path, gt_path: Path) -> dict[str, float | int | None]:
    gt_rows = {row["id"]: row for row in load_jsonl(gt_path)}
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = payload.get("questions", [])

    yes_true: list[str] = []
    yes_pred: list[str] = []
    fact_strict: list[float] = []
    fact_mrr: list[float] = []
    list_precision: list[float] = []
    list_recall: list[float] = []
    list_f1: list[float] = []
    rouge_scores: list[float] = []
    bert_candidates: list[str] = []
    bert_refs: list[str] = []
    latencies: list[float] = []

    for pred in predictions:
        gt = gt_rows.get(pred.get("id"))
        if not gt:
            continue
        answer = str(pred.get("ideal_answer", ""))
        qtype = gt.get("type")
        exact_groups = [
            {normalize_match(value) for value in group if normalize_match(value)}
            for group in flatten_exact(gt.get("exact_answer"))
        ]

        metadata = pred.get("metadata", {}) or {}
        if isinstance(metadata.get("latency"), (int, float)):
            latencies.append(float(metadata["latency"]))

        reference = " ".join(gt.get("ideal_answer", []) or [])
        if reference:
            rouge_scores.append(rouge_l(answer, reference))
            bert_candidates.append(answer)
            bert_refs.append(reference)

        if qtype == "yesno":
            gold = "unknown"
            if exact_groups and exact_groups[0]:
                gold = next(iter(exact_groups[0]))
            yes_true.append(gold)
            yes_pred.append(yesno_label(answer))
        elif qtype == "factoid":
            candidates = split_prediction_items(answer)[:5]
            first = candidates[0] if candidates else ""
            fact_strict.append(
                1.0 if any(group_hit(first, group) for group in exact_groups) else 0.0
            )
            reciprocal = 0.0
            for rank, candidate in enumerate(candidates, start=1):
                if any(group_hit(candidate, group) for group in exact_groups):
                    reciprocal = 1.0 / rank
                    break
            fact_mrr.append(reciprocal)
        elif qtype == "list":
            candidates = split_prediction_items(answer)
            matched_groups: set[int] = set()
            true_positive = 0
            for candidate in candidates:
                for index, group in enumerate(exact_groups):
                    if index in matched_groups:
                        continue
                    if group_hit(candidate, group):
                        matched_groups.add(index)
                        true_positive += 1
                        break
            precision = true_positive / len(candidates) if candidates else 0.0
            recall = len(matched_groups) / len(exact_groups) if exact_groups else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            list_precision.append(precision)
            list_recall.append(recall)
            list_f1.append(f1)

    scores: dict[str, float | int | None] = {
        "count": len(predictions),
        "yesno_accuracy": accuracy(yes_true, yes_pred),
        "yesno_macro_f1": macro_f1(yes_true, yes_pred, labels=["yes", "no"]),
        "factoid_strict_accuracy": mean(fact_strict),
        "factoid_mrr": mean(fact_mrr),
        "list_precision": mean(list_precision),
        "list_recall": mean(list_recall),
        "list_f1": mean(list_f1),
        "rouge_l": mean(rouge_scores),
        "average_latency": mean(latencies) if latencies else None,
        "bert_score": None,
    }

    try:
        from bert_score import score as bert_score

        if bert_candidates:
            _, _, f1_values = bert_score(bert_candidates, bert_refs, lang="en", verbose=False)
            scores["bert_score"] = float(f1_values.mean().item())
    except Exception as error:
        scores["bert_score_error"] = str(error)
    return scores


def gain(base: float | int | None, ours: float | int | None) -> str:
    if not isinstance(base, (int, float)) or not isinstance(ours, (int, float)) or base == 0:
        return ""
    return f"{(ours - base) / base * 100:+.1f}%"


def format_value(value: float | int | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioRAG predictions.")
    parser.add_argument("--gt", type=Path, default=EVAL_PROCESSED)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PREDICTIONS)
    parser.add_argument("--full", type=Path, default=FULL_PREDICTIONS)
    args = parser.parse_args()

    ensure_dirs()
    results: dict[str, dict[str, float | int | None]] = {}
    if args.baseline.exists():
        results["baseline"] = score_predictions(args.baseline, args.gt)
    if args.full.exists():
        results["biorag"] = score_predictions(args.full, args.gt)
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    metrics = [
        "yesno_accuracy",
        "yesno_macro_f1",
        "factoid_strict_accuracy",
        "factoid_mrr",
        "list_recall",
        "list_f1",
        "rouge_l",
        "bert_score",
        "average_latency",
    ]
    lines = [
        "# Evaluation Results",
        "",
        "| Metric | Baseline | BioRAG | Gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    baseline = results.get("baseline", {})
    biorag = results.get("biorag", {})
    for metric in metrics:
        base_value = baseline.get(metric)
        ours_value = biorag.get(metric)
        lines.append(
            f"| {metric} | {format_value(base_value)} | "
            f"{format_value(ours_value)} | {gain(base_value, ours_value)} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved JSON: {RESULTS_JSON}")
    print(f"Saved table: {RESULTS_MD}")


if __name__ == "__main__":
    main()
