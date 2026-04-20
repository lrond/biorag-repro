from __future__ import annotations

from collections import Counter
from pathlib import Path

from biorag.bioasq import parse_bioasq_questions
from biorag.config import ProjectConfig, QuestionSourceConfig
from biorag.io import dump_jsonl, write_json
from biorag.sampling import stratified_sample_questions
from biorag.types import QuestionRecord
from biorag.utils import ensure_dir


def _dedupe_questions(questions: list[QuestionRecord]) -> list[QuestionRecord]:
    ordered: list[QuestionRecord] = []
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            continue
        seen.add(question.id)
        ordered.append(question)
    return ordered


def _type_counts(questions: list[QuestionRecord]) -> dict[str, int]:
    return dict(Counter(question.type for question in questions))


def _load_question_source(
    source: QuestionSourceConfig,
    *,
    fallback_path: str = "",
) -> list[QuestionRecord]:
    source_path = source.path or fallback_path
    if not source_path:
        return []
    kwargs: dict[str, object] = {}
    if source.member_names:
        kwargs["member_names"] = source.member_names
    elif source.member_glob:
        kwargs["member_glob"] = source.member_glob
    return parse_bioasq_questions(source_path, **kwargs)


def materialize_question_splits(
    config: ProjectConfig,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_root = ensure_dir(output_dir)
    protocol = config.dataset.protocol
    training_questions = _dedupe_questions(
        _load_question_source(
            config.dataset.training,
            fallback_path=config.dataset.question_path,
        )
    )
    golden_questions = _dedupe_questions(_load_question_source(config.dataset.golden))
    if not training_questions:
        raise ValueError("No BioASQ training questions were resolved from the configured dataset source.")

    notes: list[str] = []
    if protocol == "single_file":
        all_questions = list(training_questions)
        train_questions = list(training_questions)
        evaluation_questions = stratified_sample_questions(
            training_questions,
            sample_size=config.dataset.sample_size,
            seed=config.dataset.sample_seed,
        )
    elif protocol == "report_holdout":
        if config.dataset.sample_size >= len(training_questions):
            raise ValueError(
                "The report_holdout protocol requires sample_size to be smaller than the number of training questions."
            )
        all_questions = list(training_questions)
        evaluation_questions = stratified_sample_questions(
            training_questions,
            sample_size=config.dataset.sample_size,
            seed=config.dataset.sample_seed,
        )
        evaluation_ids = {question.id for question in evaluation_questions}
        train_questions = [question for question in training_questions if question.id not in evaluation_ids]
        notes.append(
            "Evaluation split follows the report protocol: a stratified random holdout from BioASQ-training12b "
            f"(N={len(evaluation_questions)}, seed={config.dataset.sample_seed}) with zero question overlap."
        )
    elif protocol == "golden_eval":
        if not golden_questions:
            raise ValueError("golden_eval protocol requires a configured golden dataset source.")
        train_questions = list(training_questions)
        evaluation_questions = list(golden_questions)
        all_questions = _dedupe_questions(training_questions + golden_questions)
        notes.append(
            "Evaluation split uses the official BioASQ 12b golden-enriched batches while training uses BioASQ-training12b."
        )
    else:
        raise ValueError(f"Unsupported dataset protocol: {protocol}")

    all_path = dump_jsonl(output_root / "all_questions.jsonl", all_questions)
    train_path = dump_jsonl(output_root / "train_questions.jsonl", train_questions)
    evaluation_path = dump_jsonl(output_root / "evaluation_questions.jsonl", evaluation_questions)
    artifacts: dict[str, Path] = {
        "all": all_path,
        "canonical": all_path,
        "train": train_path,
        "evaluation": evaluation_path,
        "sampled": evaluation_path,
    }
    if golden_questions:
        artifacts["golden"] = dump_jsonl(output_root / "golden_questions.jsonl", golden_questions)
    manifest = {
        "protocol": protocol,
        "seed": config.dataset.sample_seed,
        "requested_sample_size": config.dataset.sample_size,
        "all_count": len(all_questions),
        "train_count": len(train_questions),
        "evaluation_count": len(evaluation_questions),
        "golden_count": len(golden_questions),
        "all_type_distribution": _type_counts(all_questions),
        "train_type_distribution": _type_counts(train_questions),
        "evaluation_type_distribution": _type_counts(evaluation_questions),
        "golden_type_distribution": _type_counts(golden_questions),
        "evaluation_ids": [question.id for question in evaluation_questions],
        "notes": notes,
    }
    artifacts["manifest"] = write_json(output_root / "split_manifest.json", manifest)
    return artifacts
