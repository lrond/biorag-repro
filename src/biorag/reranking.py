from __future__ import annotations

import time
from pathlib import Path

from biorag.config import ProjectConfig, resolve_model_source
from biorag.io import dump_jsonl
from biorag.types import QuestionRecord, RetrievedContext, ScoredDocument
from biorag.utils import get_logger, simple_tokenize

LOGGER = get_logger(__name__)


def _lexical_rerank_score(question_text: str, document_text: str) -> float:
    query_tokens = set(simple_tokenize(question_text))
    document_tokens = set(simple_tokenize(document_text))
    overlap = len(query_tokens & document_tokens)
    exact_phrase_bonus = 0.5 if question_text.lower() in document_text.lower() else 0.0
    return float(overlap + exact_phrase_bonus)


def rerank_contexts(
    questions: list[QuestionRecord],
    contexts: list[RetrievedContext],
    config: ProjectConfig,
    output_path: str | Path,
    device: str,
) -> Path:
    if not config.inference.rerank_enabled or config.models.reranker.backend == "disabled":
        passthrough = [
            context.model_copy(
                update={
                    "stage": "rerank_skipped",
                    "candidates": context.candidates[: config.inference.final_top_k],
                    "metadata": {**context.metadata, "rerank_latency_seconds": 0.0},
                }
            )
            for context in contexts
        ]
        return dump_jsonl(output_path, passthrough)
    questions_by_id = {question.id: question for question in questions}
    reranker_backend = config.models.reranker.backend
    reranked: list[RetrievedContext] = []
    if reranker_backend == "lexical":
        for context in contexts:
            start = time.perf_counter()
            question = questions_by_id[context.question_id]
            ranked = sorted(
                context.candidates[: config.inference.rerank_top_k],
                key=lambda candidate: _lexical_rerank_score(question.body, candidate.text),
                reverse=True,
            )[: config.inference.final_top_k]
            reranked.append(
                RetrievedContext(
                    question_id=context.question_id,
                    question_type=context.question_type,
                    question=context.question,
                    stage="reranked",
                    metadata={
                        **context.metadata,
                        "rerank_latency_seconds": time.perf_counter() - start,
                    },
                    candidates=[
                        candidate.model_copy(update={"rank": index + 1})
                        for index, candidate in enumerate(ranked)
                    ],
                )
            )
        return dump_jsonl(output_path, reranked)
    if reranker_backend != "cross_encoder":
        raise ValueError(f"Unsupported reranker backend: {reranker_backend}")
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sentence-transformers is required for cross-encoder reranking."
        ) from error
    model_source = resolve_model_source(
        config.models.reranker.model_name,
        checkpoint_path=config.models.reranker.checkpoint_path,
    )
    model = CrossEncoder(model_source, device=device)
    for context in contexts:
        start = time.perf_counter()
        question = questions_by_id[context.question_id]
        pairs = [
            (question.body, candidate.text)
            for candidate in context.candidates[: config.inference.rerank_top_k]
        ]
        scores = model.predict(pairs) if pairs else []
        rescored = []
        for candidate, score in zip(context.candidates[: config.inference.rerank_top_k], scores):
            rescored.append(candidate.model_copy(update={"score": float(score)}))
        ranked = sorted(rescored, key=lambda candidate: candidate.score, reverse=True)[
            : config.inference.final_top_k
        ]
        reranked.append(
            RetrievedContext(
                question_id=context.question_id,
                question_type=context.question_type,
                question=context.question,
                stage="reranked",
                metadata={
                    **context.metadata,
                    "rerank_latency_seconds": time.perf_counter() - start,
                },
                candidates=[
                    ScoredDocument(**candidate.model_dump(mode="python"), rank=index + 1)
                    for index, candidate in enumerate(ranked)
                ],
            )
        )
    return dump_jsonl(output_path, reranked)
