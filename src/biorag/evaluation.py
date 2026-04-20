from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from biorag.config import ProjectConfig
from biorag.io import write_json, write_text
from biorag.types import EvalReport, PredictionRecord, QuestionRecord
from biorag.utils import normalize_text


def _flatten_gold_answers(answer_groups: list[list[str]]) -> set[str]:
    flattened: set[str] = set()
    for group in answer_groups:
        flattened.update(normalize_text(item) for item in group if item.strip())
    return {item for item in flattened if item}


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
    factoid_mrr: list[float] = []
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
            gold = question.exact_answer[0][0] if question.exact_answer and question.exact_answer[0] else "unknown"
            yesno_gold.append(normalize_text(gold))
            yesno_pred.append(normalize_text(prediction.answer))
        elif question.type == "factoid":
            gold_groups = question.exact_answer
            normalized_gold = [normalize_text(item) for group in gold_groups for item in group]
            ranked = [normalize_text(item) for item in (prediction.ranked_answers or [prediction.answer])]
            strict_hit = 1.0 if ranked and any(ranked[0] == answer for answer in normalized_gold) else 0.0
            reciprocal_rank = 0.0
            for rank, candidate in enumerate(ranked, start=1):
                if candidate in normalized_gold:
                    reciprocal_rank = 1.0 / rank
                    break
            factoid_sacc.append(strict_hit)
            factoid_mrr.append(reciprocal_rank)
        elif question.type == "list":
            gold_groups = question.exact_answer
            gold_flat = _flatten_gold_answers(gold_groups)
            pred_flat = {normalize_text(item) for item in prediction.ranked_answers if item.strip()}
            true_positive = len(gold_flat & pred_flat)
            recall = true_positive / len(gold_flat) if gold_flat else 0.0
            precision = true_positive / len(pred_flat) if pred_flat else 0.0
            f_measure = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            list_recall.append(recall)
            list_fmeasure.append(f_measure)
        ideal_reference = " ".join(question.ideal_answer).strip()
        if ideal_reference:
            rouge_scores.append(_rouge_l_score(prediction.answer, ideal_reference))
            bert_predictions.append(prediction.answer)
            bert_references.append(ideal_reference)

    if yesno_gold:
        overall_metrics["yesno_accuracy"] = sum(g == p for g, p in zip(yesno_gold, yesno_pred)) / len(yesno_gold)
        overall_metrics["yesno_macro_f1"] = _macro_f1(yesno_gold, yesno_pred)
        per_type_metrics["yesno"]["accuracy"] = overall_metrics["yesno_accuracy"]
        per_type_metrics["yesno"]["macro_f1"] = overall_metrics["yesno_macro_f1"]
    if factoid_sacc:
        overall_metrics["factoid_strict_accuracy"] = _mean(factoid_sacc)
        overall_metrics["factoid_mrr"] = _mean(factoid_mrr)
        per_type_metrics["factoid"]["strict_accuracy"] = overall_metrics["factoid_strict_accuracy"]
        per_type_metrics["factoid"]["mrr"] = overall_metrics["factoid_mrr"]
    if list_recall:
        overall_metrics["list_recall"] = _mean(list_recall)
        overall_metrics["list_fmeasure"] = _mean(list_fmeasure)
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
    report_path = write_json(output_root / config.evaluation.report_name, report.model_dump(mode="json"))
    summary_path = write_text(output_root / config.evaluation.summary_table_name, render_summary_table(report))
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
    if report.notes:
        lines.extend(["", "## Notes", ""])
        for note in report.notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"
