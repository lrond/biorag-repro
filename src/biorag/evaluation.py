from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from biorag.config import ProjectConfig
from biorag.io import write_json, write_text
from biorag.types import EvalReport, PredictionRecord, QuestionRecord
from biorag.utils import normalize_text


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _lcs_length(a: list[str], b: list[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def _rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _maybe_bert_score(predictions: list[str], references: list[str]) -> float | None:
    try:
        from bert_score import score  # type: ignore
    except ModuleNotFoundError:
        return None
    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return float(f1.mean().item())


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _normalize_gold_groups(answer_groups: list[list[str]]) -> list[set[str]]:
    groups: list[set[str]] = []
    for group in answer_groups:
        normalized = {normalize_text(item) for item in group if normalize_text(item)}
        if normalized:
            groups.append(normalized)
    return groups


def _match_groups(predictions: list[str], gold_groups: list[set[str]]) -> tuple[int, int]:
    matched_groups: set[int] = set()
    matched_predictions = 0
    for prediction in predictions:
        for index, group in enumerate(gold_groups):
            if index in matched_groups:
                continue
            if prediction in group:
                matched_groups.add(index)
                matched_predictions += 1
                break
    return matched_predictions, len(matched_groups)


def evaluate_predictions(
    questions: list[QuestionRecord],
    predictions: list[PredictionRecord],
    config: ProjectConfig,
    output_dir: str | Path,
) -> tuple[Path, Path, EvalReport]:
    prediction_by_id = {prediction.question_id: prediction for prediction in predictions}
    counts = Counter(question.type for question in questions)
    notes: list[str] = []
    per_type_metrics: dict[str, dict[str, float | None]] = defaultdict(dict)
    overall_metrics: dict[str, float | None] = {}

    yesno_gold: list[str] = []
    yesno_pred: list[str] = []
    factoid_sacc: list[float] = []
    factoid_lenient: list[float] = []
    factoid_mrr: list[float] = []
    list_precision: list[float] = []
    list_recall: list[float] = []
    list_fmeasure: list[float] = []
    rouge_scores: list[float] = []
    bert_predictions: list[str] = []
    bert_references: list[str] = []
    latencies: list[float] = []

    for question in questions:
        prediction = prediction_by_id.get(question.id)
        if prediction is None:
            prediction = PredictionRecord(
                question_id=question.id,
                question_type=question.type,
                answer="I don't know.",
                abstained=True,
            )
        if prediction.latency_seconds is not None:
            latencies.append(prediction.latency_seconds)
        if question.type == "yesno":
            gold = (
                question.exact_answer[0][0]
                if question.exact_answer and question.exact_answer[0]
                else "unknown"
            )
            yesno_gold.append(normalize_text(gold))
            yesno_pred.append(normalize_text(prediction.answer))
        elif question.type == "factoid":
            gold_groups = _normalize_gold_groups(question.exact_answer)
            ranked = _dedupe_preserving_order(
                [
                    normalize_text(item)
                    for item in (prediction.ranked_answers or [prediction.answer])
                ]
            )[:5]
            strict_hit = 1.0 if ranked and any(ranked[0] in group for group in gold_groups) else 0.0
            lenient_hit = (
                1.0
                if any(any(candidate in group for group in gold_groups) for candidate in ranked)
                else 0.0
            )
            reciprocal_rank = 0.0
            for rank, candidate in enumerate(ranked, start=1):
                if any(candidate in group for group in gold_groups):
                    reciprocal_rank = 1.0 / rank
                    break
            factoid_sacc.append(strict_hit)
            factoid_lenient.append(lenient_hit)
            factoid_mrr.append(reciprocal_rank)
        elif question.type == "list":
            gold_groups = _normalize_gold_groups(question.exact_answer)
            ranked_predictions = prediction.ranked_answers or [prediction.answer]
            pred_flat = _dedupe_preserving_order(
                [normalize_text(item) for item in ranked_predictions if item.strip()]
            )
            true_positive, matched_groups = _match_groups(pred_flat, gold_groups)
            recall = matched_groups / len(gold_groups) if gold_groups else 0.0
            precision = true_positive / len(pred_flat) if pred_flat else 0.0
            f_measure = (
                (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            )
            list_precision.append(precision)
            list_recall.append(recall)
            list_fmeasure.append(f_measure)
        ideal_reference = " ".join(question.ideal_answer).strip()
        if ideal_reference:
            rouge_scores.append(_rouge_l_score(prediction.answer, ideal_reference))
            bert_predictions.append(prediction.answer)
            bert_references.append(ideal_reference)

    if yesno_gold:
        overall_metrics["yesno_accuracy"] = sum(
            g == p for g, p in zip(yesno_gold, yesno_pred)
        ) / len(yesno_gold)
        for label in ("yes", "no"):
            tp = sum(1 for g, p in zip(yesno_gold, yesno_pred) if g == label and p == label)
            fp = sum(1 for g, p in zip(yesno_gold, yesno_pred) if g != label and p == label)
            fn = sum(1 for g, p in zip(yesno_gold, yesno_pred) if g == label and p != label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            overall_metrics[f"yesno_f1_{label}"] = f1
            per_type_metrics["yesno"][f"f1_{label}"] = f1
        overall_metrics["yesno_macro_f1"] = _macro_f1(yesno_gold, yesno_pred)
        per_type_metrics["yesno"]["accuracy"] = overall_metrics["yesno_accuracy"]
        per_type_metrics["yesno"]["macro_f1"] = overall_metrics["yesno_macro_f1"]
    if factoid_sacc:
        overall_metrics["factoid_strict_accuracy"] = _mean(factoid_sacc)
        overall_metrics["factoid_lenient_accuracy"] = _mean(factoid_lenient)
        overall_metrics["factoid_mrr"] = _mean(factoid_mrr)
        per_type_metrics["factoid"]["strict_accuracy"] = overall_metrics["factoid_strict_accuracy"]
        per_type_metrics["factoid"]["lenient_accuracy"] = overall_metrics[
            "factoid_lenient_accuracy"
        ]
        per_type_metrics["factoid"]["mrr"] = overall_metrics["factoid_mrr"]
    if list_recall:
        overall_metrics["list_mean_precision"] = _mean(list_precision)
        overall_metrics["list_recall"] = _mean(list_recall)
        overall_metrics["list_fmeasure"] = _mean(list_fmeasure)
        per_type_metrics["list"]["mean_precision"] = overall_metrics["list_mean_precision"]
        per_type_metrics["list"]["recall"] = overall_metrics["list_recall"]
        per_type_metrics["list"]["fmeasure"] = overall_metrics["list_fmeasure"]
    if config.evaluation.compute_rouge_l and rouge_scores:
        overall_metrics["rouge_l"] = _mean(rouge_scores)
    if config.evaluation.compute_bert_score and bert_predictions:
        bert_score_value = _maybe_bert_score(bert_predictions, bert_references)
        overall_metrics["bert_score"] = bert_score_value
        if bert_score_value is None:
            notes.append("bert-score dependency not installed; bert_score metric set to null.")
    if config.evaluation.compute_latency and latencies:
        overall_metrics["average_latency_seconds"] = _mean(latencies)

    report = EvalReport(
        metrics=overall_metrics,
        per_type_metrics=dict(per_type_metrics),
        counts=dict(counts),
        notes=notes,
    )
    output_root = Path(output_dir)
    report_path = write_json(
        output_root / config.evaluation.report_name, report.model_dump(mode="json")
    )
    summary_path = write_text(
        output_root / config.evaluation.summary_table_name, render_summary_table(report)
    )
    return report_path, summary_path, report


def render_summary_table(report: EvalReport) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric, value in sorted(report.metrics.items()):
        formatted = "null" if value is None else f"{value:.4f}"
        lines.append(f"| {metric} | {formatted} |")
    if report.per_type_metrics:
        lines.extend(["", "## Per-type Metrics", ""])
        for question_type, metrics in sorted(report.per_type_metrics.items()):
            lines.append(f"### {question_type}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")
            for metric, value in sorted(metrics.items()):
                formatted = "null" if value is None else f"{value:.4f}"
                lines.append(f"| {metric} | {formatted} |")
            lines.append("")
    if report.notes:
        lines.extend(["", "## Notes", ""])
        for note in report.notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"
