from __future__ import annotations

from pathlib import Path

from biorag.bioasq import parse_bioasq_questions
from biorag.config import ProjectConfig
from biorag.corpus import (
    build_corpus_lookup,
    build_linked_pubmed_corpus,
    build_pubmed_dump_corpus,
    load_corpus,
)
from biorag.evaluation import evaluate_predictions
from biorag.generation import generate_predictions
from biorag.io import dump_jsonl, load_jsonl, write_json
from biorag.reranking import rerank_contexts
from biorag.retrieval import build_index, retrieve_questions, train_contrastive_retriever
from biorag.sampling import stratified_sample_questions
from biorag.types import PredictionRecord, QuestionRecord, RetrievedContext
from biorag.utils import ensure_dir


def _run_root(config: ProjectConfig) -> Path:
    return ensure_dir(Path(config.output_dir) / config.run_name)


def save_config_snapshot(config: ProjectConfig) -> Path:
    run_root = _run_root(config)
    return write_json(run_root / "resolved_config.json", config.model_dump(mode="json"))


def _question_input_path(config: ProjectConfig) -> Path:
    path = Path(config.dataset.question_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def prepare_data(config: ProjectConfig) -> dict[str, Path]:
    run_root = _run_root(config)
    canonical_dir = ensure_dir(run_root / "canonical")
    questions = parse_bioasq_questions(_question_input_path(config))
    sampled = stratified_sample_questions(
        questions,
        sample_size=config.dataset.sample_size,
        seed=config.dataset.sample_seed,
    )
    canonical_path = dump_jsonl(canonical_dir / "questions.jsonl", questions)
    sampled_path = dump_jsonl(canonical_dir / "sampled_questions.jsonl", sampled)
    sample_manifest = write_json(
        canonical_dir / "sample_manifest.json",
        {
            "sample_size": len(sampled),
            "seed": config.dataset.sample_seed,
            "question_ids": [question.id for question in sampled],
        },
    )
    return {"canonical": canonical_path, "sampled": sampled_path, "manifest": sample_manifest}


def build_corpus_stage(config: ProjectConfig) -> dict[str, Path]:
    run_root = _run_root(config)
    canonical_paths = prepare_data(config)
    questions = [QuestionRecord.model_validate(row) for row in load_jsonl(canonical_paths["canonical"])]
    corpus_dir = ensure_dir(run_root / "corpus")
    corpus_path = corpus_dir / "corpus.jsonl"
    if config.dataset.corpus_mode == "linked_pubmed":
        output_path, manifest_path = build_linked_pubmed_corpus(questions, config.dataset, corpus_path)
    elif config.dataset.corpus_mode == "pubmed_dump":
        output_path, manifest_path = build_pubmed_dump_corpus(questions, config.dataset, corpus_path)
    else:
        raise ValueError(f"Unsupported corpus mode: {config.dataset.corpus_mode}")
    return {"corpus": output_path, "manifest": manifest_path, **canonical_paths}


def build_index_stage(config: ProjectConfig, model_source: str | None = None) -> dict[str, Path]:
    run_root = _run_root(config)
    corpus_paths = build_corpus_stage(config)
    documents = load_corpus(corpus_paths["corpus"])
    index_dir = ensure_dir(run_root / "index")
    build_index(documents, config, index_dir, device=config.device, model_source=model_source)
    return {"index": index_dir, **corpus_paths}


def train_retriever_stage(config: ProjectConfig) -> dict[str, Path]:
    run_root = _run_root(config)
    corpus_paths = build_corpus_stage(config)
    questions = [QuestionRecord.model_validate(row) for row in load_jsonl(corpus_paths["canonical"])]
    documents = load_corpus(corpus_paths["corpus"])
    training_dir = ensure_dir(run_root / "training")
    model_path = train_contrastive_retriever(questions, documents, config, training_dir, device=config.device)
    return {"model": Path(model_path), **corpus_paths}


def retrieve_stage(config: ProjectConfig, model_source: str | None = None) -> dict[str, Path]:
    run_root = _run_root(config)
    index_paths = build_index_stage(config, model_source=model_source)
    documents = load_corpus(index_paths["corpus"])
    questions = [QuestionRecord.model_validate(row) for row in load_jsonl(index_paths["sampled"])]
    retrieval_dir = ensure_dir(run_root / "retrieval")
    output_path = retrieve_questions(
        questions,
        documents,
        config,
        index_paths["index"],
        retrieval_dir / "retrieved_contexts.jsonl",
        device=config.device,
        model_source=model_source,
    )
    return {"retrieved": output_path, **index_paths}


def rerank_stage(config: ProjectConfig) -> dict[str, Path]:
    run_root = _run_root(config)
    retrieval_paths = retrieve_stage(config)
    questions = [QuestionRecord.model_validate(row) for row in load_jsonl(retrieval_paths["sampled"])]
    contexts = [RetrievedContext.model_validate(row) for row in load_jsonl(retrieval_paths["retrieved"])]
    rerank_dir = ensure_dir(run_root / "rerank")
    output_path = rerank_contexts(
        questions,
        contexts,
        config,
        rerank_dir / "reranked_contexts.jsonl",
        device=config.device,
    )
    return {"reranked": output_path, **retrieval_paths}


def generate_stage(config: ProjectConfig) -> dict[str, Path]:
    run_root = _run_root(config)
    if config.inference.rerank_enabled:
        stage_paths = rerank_stage(config)
        context_path = stage_paths["reranked"]
    else:
        stage_paths = retrieve_stage(config)
        context_path = stage_paths["retrieved"]
    questions = [QuestionRecord.model_validate(row) for row in load_jsonl(stage_paths["sampled"])]
    contexts = [RetrievedContext.model_validate(row) for row in load_jsonl(context_path)]
    prediction_dir = ensure_dir(run_root / "predictions")
    output_path = generate_predictions(
        questions,
        contexts,
        config,
        prediction_dir / "prediction.jsonl",
        device=config.device,
    )
    return {"predictions": output_path, **stage_paths}


def evaluate_stage(config: ProjectConfig) -> dict[str, Path]:
    run_root = _run_root(config)
    generation_paths = generate_stage(config)
    questions = [QuestionRecord.model_validate(row) for row in load_jsonl(generation_paths["sampled"])]
    predictions = [PredictionRecord.model_validate(row) for row in load_jsonl(generation_paths["predictions"])]
    evaluation_dir = ensure_dir(run_root / "evaluation")
    report_path, summary_path, report = evaluate_predictions(questions, predictions, config, evaluation_dir)
    artifacts = {
        "report": report_path,
        "summary": summary_path,
    }
    report_payload = report.model_dump(mode="json")
    report_payload["artifacts"] = {key: str(value) for key, value in artifacts.items()}
    write_json(report_path, report_payload)
    return {**generation_paths, **artifacts}


def run_baseline(config: ProjectConfig) -> dict[str, Path]:
    config = config.model_copy(
        update={
            "inference": config.inference.model_copy(update={"rerank_enabled": False, "retrieve_top_k": 3, "final_top_k": 3})
        }
    )
    return evaluate_stage(config)


def run_full_pipeline(config: ProjectConfig) -> dict[str, Path]:
    training_paths = train_retriever_stage(config)
    trained_model = training_paths["model"]
    index_paths = build_index_stage(config, model_source=str(trained_model))
    questions = [QuestionRecord.model_validate(row) for row in load_jsonl(index_paths["sampled"])]
    documents = load_corpus(index_paths["corpus"])
    retrieval_dir = ensure_dir(_run_root(config) / "retrieval")
    retrieved_path = retrieve_questions(
        questions,
        documents,
        config,
        index_paths["index"],
        retrieval_dir / "retrieved_contexts.jsonl",
        device=config.device,
        model_source=str(trained_model),
    )
    contexts = [RetrievedContext.model_validate(row) for row in load_jsonl(retrieved_path)]
    rerank_dir = ensure_dir(_run_root(config) / "rerank")
    reranked_path = rerank_contexts(
        questions,
        contexts,
        config,
        rerank_dir / "reranked_contexts.jsonl",
        device=config.device,
    )
    final_contexts = [RetrievedContext.model_validate(row) for row in load_jsonl(reranked_path)]
    prediction_dir = ensure_dir(_run_root(config) / "predictions")
    prediction_path = generate_predictions(
        questions,
        final_contexts,
        config,
        prediction_dir / "prediction.jsonl",
        device=config.device,
    )
    predictions = [PredictionRecord.model_validate(row) for row in load_jsonl(prediction_path)]
    evaluation_dir = ensure_dir(_run_root(config) / "evaluation")
    report_path, summary_path, report = evaluate_predictions(questions, predictions, config, evaluation_dir)
    report_payload = report.model_dump(mode="json")
    report_payload["artifacts"] = {
        "trained_model": str(trained_model),
        "retrieved": str(retrieved_path),
        "reranked": str(reranked_path),
        "predictions": str(prediction_path),
        "report": str(report_path),
        "summary": str(summary_path),
    }
    write_json(report_path, report_payload)
    return {
        "trained_model": trained_model,
        "retrieved": retrieved_path,
        "reranked": reranked_path,
        "predictions": prediction_path,
        "report": report_path,
        "summary": summary_path,
    }
