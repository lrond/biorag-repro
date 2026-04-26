from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from biorag.bioasq import build_training_pairs
from biorag.config import ProjectConfig, resolve_model_source
from biorag.io import dump_jsonl, load_jsonl, write_json
from biorag.modeling import autocast_dtype, model_load_kwargs
from biorag.types import DocumentRecord, QuestionRecord, RetrievedContext, ScoredDocument
from biorag.utils import batched, ensure_dir, get_logger, simple_tokenize

LOGGER = get_logger(__name__)


def _numpy():
    import numpy as np  # type: ignore

    return np


def _transformers():
    import torch  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    return torch, AutoModel, AutoTokenizer


def _faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except ModuleNotFoundError:
        return None


def _pool_embeddings(last_hidden_state: Any, attention_mask: Any, pooling: str) -> Any:
    if pooling == "mean":
        masked = last_hidden_state * attention_mask.unsqueeze(-1)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return masked.sum(dim=1) / denom
    return last_hidden_state[:, 0]


def _normalize_matrix(matrix: Any) -> Any:
    torch, _, _ = _transformers()
    return torch.nn.functional.normalize(matrix, p=2, dim=1)


def _use_cuda_autocast(device: str, dtype: Any) -> bool:
    return dtype is not None and str(device).split(":")[0] == "cuda"


def _lexical_score(query: str, document: str) -> float:
    query_tokens = set(simple_tokenize(query))
    document_tokens = set(simple_tokenize(document))
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(document_tokens))


def _search_faiss_index(index_dir: Path, queries: Any, top_k: int) -> tuple[Any, Any] | None:
    if top_k <= 0:
        return None
    faiss_module = _faiss()
    faiss_path = index_dir / "faiss.index"
    if faiss_module is None or not faiss_path.exists():
        return None
    index = faiss_module.read_index(str(faiss_path))
    return index.search(queries.astype("float32"), top_k)


def _should_step_optimizer(batch_index: int, total_batches: int, accumulation_steps: int) -> bool:
    return batch_index % max(accumulation_steps, 1) == 0 or batch_index == total_batches


def build_index(
    documents: list[DocumentRecord],
    config: ProjectConfig,
    output_dir: str | Path,
    device: str,
    model_source: str | None = None,
) -> Path:
    index_dir = ensure_dir(output_dir)
    backend = config.models.retriever.backend
    if backend == "lexical":
        payload = [
            {
                "id": document.id,
                "title": document.title,
                "text": document.text,
                "tokens": simple_tokenize(document.text),
            }
            for document in documents
        ]
        dump_jsonl(index_dir / "lexical_index.jsonl", payload)
        write_json(
            index_dir / "index_metadata.json",
            {"backend": backend, "document_count": len(documents)},
        )
        return index_dir
    if backend != "transformer":
        raise ValueError(f"Unsupported retriever backend: {backend}")
    np = _numpy()
    torch, AutoModel, AutoTokenizer = _transformers()
    source = resolve_model_source(
        config.models.retriever.model_name,
        mode=config.models.retriever.mode,
        checkpoint_path=config.models.retriever.checkpoint_path,
        override_path=model_source,
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(
        source,
        **model_load_kwargs(config.models.retriever, torch),
    )
    model.to(device)
    model.eval()
    embeddings = []
    doc_ids = []
    amp_dtype = autocast_dtype(config.training.mixed_precision, torch)
    use_amp = _use_cuda_autocast(device, amp_dtype)
    with torch.no_grad():
        for batch in batched(documents, config.training.batch_size):
            texts = [document.text for document in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.training.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**encoded)
                pooled = _pool_embeddings(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                    config.models.retriever.pooling,
                )
                if config.models.retriever.normalize:
                    pooled = _normalize_matrix(pooled)
            embeddings.append(pooled.float().cpu().numpy())
            doc_ids.extend(document.id for document in batch)
    stacked = np.concatenate(embeddings, axis=0)
    np.save(index_dir / "embeddings.npy", stacked)
    dump_jsonl(index_dir / "documents.jsonl", documents)
    write_json(index_dir / "doc_ids.json", doc_ids)
    faiss_module = _faiss()
    if faiss_module is not None:
        index = faiss_module.IndexFlatIP(stacked.shape[1])
        index.add(stacked.astype("float32"))
        faiss_module.write_index(index, str(index_dir / "faiss.index"))
    write_json(
        index_dir / "index_metadata.json",
        {"backend": backend, "document_count": len(documents), "model_source": source},
    )
    return index_dir


def _load_transformer_documents(index_dir: Path) -> tuple[list[DocumentRecord], list[str]]:
    documents = [
        DocumentRecord.model_validate(row) for row in load_jsonl(index_dir / "documents.jsonl")
    ]
    ids = [document.id for document in documents]
    return documents, ids


def _load_transformer_embeddings(index_dir: Path) -> Any:
    np = _numpy()
    return np.load(index_dir / "embeddings.npy")


def retrieve_questions(
    questions: list[QuestionRecord],
    documents: list[DocumentRecord],
    config: ProjectConfig,
    index_dir: str | Path,
    output_path: str | Path,
    device: str,
    model_source: str | None = None,
) -> Path:
    backend = config.models.retriever.backend
    documents_by_id = {document.id: document for document in documents}
    contexts: list[RetrievedContext] = []
    if backend == "lexical":
        start = time.perf_counter()
        tokenized_docs = {document.id: document.text for document in documents}
        for question in questions:
            ranked = sorted(
                (
                    (document_id, _lexical_score(question.body, text))
                    for document_id, text in tokenized_docs.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )[: config.inference.retrieve_top_k]
            contexts.append(
                RetrievedContext(
                    question_id=question.id,
                    question_type=question.type,
                    question=question.body,
                    stage="retrieval",
                    metadata={
                        "top_retrieval_score": float(ranked[0][1]) if ranked else 0.0,
                    },
                    candidates=[
                        ScoredDocument(
                            document_id=document_id,
                            score=float(score),
                            rank=rank + 1,
                            title=documents_by_id[document_id].title,
                            text=documents_by_id[document_id].text,
                            metadata={"retrieval_score": float(score)},
                        )
                        for rank, (document_id, score) in enumerate(ranked)
                    ],
                )
            )
        average_latency = (time.perf_counter() - start) / max(len(questions), 1)
        contexts = [
            context.model_copy(
                update={
                    "metadata": {
                        **context.metadata,
                        "retrieval_latency_seconds": average_latency,
                    }
                }
            )
            for context in contexts
        ]
        return dump_jsonl(output_path, contexts)
    if backend != "transformer":
        raise ValueError(f"Unsupported retriever backend: {backend}")
    np = _numpy()
    torch, AutoModel, AutoTokenizer = _transformers()
    index_path = Path(index_dir)
    docs, doc_ids = _load_transformer_documents(index_path)
    documents_by_id = {document.id: document for document in docs}
    if not questions:
        return dump_jsonl(output_path, [])
    if not doc_ids:
        return dump_jsonl(
            output_path,
            [
                RetrievedContext(
                    question_id=question.id,
                    question_type=question.type,
                    question=question.body,
                    stage="retrieval",
                    metadata={"retrieval_latency_seconds": 0.0},
                    candidates=[],
                )
                for question in questions
            ],
        )
    source = resolve_model_source(
        config.models.retriever.model_name,
        mode=config.models.retriever.mode,
        checkpoint_path=config.models.retriever.checkpoint_path,
        override_path=model_source,
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(
        source,
        **model_load_kwargs(config.models.retriever, torch),
    )
    model.to(device)
    model.eval()
    start = time.perf_counter()
    question_embeddings = []
    amp_dtype = autocast_dtype(config.training.mixed_precision, torch)
    use_amp = _use_cuda_autocast(device, amp_dtype)
    with torch.no_grad():
        for batch in batched(questions, config.training.batch_size):
            encoded = tokenizer(
                [question.body for question in batch],
                padding=True,
                truncation=True,
                max_length=config.training.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**encoded)
                pooled = _pool_embeddings(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                    config.models.retriever.pooling,
                )
                if config.models.retriever.normalize:
                    pooled = _normalize_matrix(pooled)
            question_embeddings.append(pooled.float().cpu().numpy())
    queries = np.concatenate(question_embeddings, axis=0)
    average_latency = (time.perf_counter() - start) / max(len(questions), 1)
    top_k = min(config.inference.retrieve_top_k, len(doc_ids))
    faiss_result = _search_faiss_index(index_path, queries, top_k)
    if faiss_result is None:
        doc_embeddings = _load_transformer_embeddings(index_path)
        scores = queries @ doc_embeddings.T
        rankings_by_question = [
            [
                (int(doc_index), float(scores[row_index][int(doc_index)]))
                for doc_index in scores[row_index].argsort()[::-1][:top_k]
            ]
            for row_index in range(len(questions))
        ]
    else:
        score_rows, index_rows = faiss_result
        rankings_by_question = [
            [
                (int(doc_index), float(score))
                for score, doc_index in zip(score_rows[row_index], index_rows[row_index])
                if int(doc_index) >= 0
            ]
            for row_index in range(len(questions))
        ]
    for row_index, question in enumerate(questions):
        ranking = rankings_by_question[row_index]
        candidates = []
        for rank, (doc_index, score) in enumerate(ranking):
            document_id = doc_ids[doc_index]
            document = documents_by_id[document_id]
            candidates.append(
                ScoredDocument(
                    document_id=document_id,
                    score=score,
                    rank=rank + 1,
                    title=document.title,
                    text=document.text,
                    metadata={"retrieval_score": score},
                )
            )
        contexts.append(
            RetrievedContext(
                question_id=question.id,
                question_type=question.type,
                question=question.body,
                stage="retrieval",
                metadata={
                    "retrieval_latency_seconds": average_latency,
                    "top_retrieval_score": candidates[0].score if candidates else 0.0,
                },
                candidates=candidates,
            )
        )
    return dump_jsonl(output_path, contexts)


def train_contrastive_retriever(
    questions: list[QuestionRecord],
    documents: list[DocumentRecord],
    config: ProjectConfig,
    output_dir: str | Path,
    device: str,
) -> Path:
    run_dir = ensure_dir(output_dir)
    if not config.training.enabled:
        metadata_path = write_json(
            run_dir / "training_metrics.json",
            {"enabled": False, "reason": "training disabled for this config"},
        )
        return metadata_path.parent
    if config.models.retriever.backend != "transformer":
        raise ValueError("Contrastive training is only implemented for transformer retrievers.")
    torch, AutoModel, AutoTokenizer = _transformers()

    corpus_by_id = {document.id: document.text for document in documents}
    pairs = build_training_pairs(questions, corpus_by_id)
    if not pairs:
        raise ValueError("No training pairs could be built from the provided questions and corpus.")
    tokenizer = AutoTokenizer.from_pretrained(config.models.retriever.model_name)
    model = AutoModel.from_pretrained(
        config.models.retriever.model_name,
        **model_load_kwargs(config.models.retriever, torch),
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    amp_dtype = autocast_dtype(config.training.mixed_precision, torch)
    use_amp = _use_cuda_autocast(device, amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    accumulation_steps = max(1, config.training.gradient_accumulation_steps)
    history: list[dict[str, float | int]] = []
    step_history: list[dict[str, float | int]] = []
    global_batch_step = 0
    global_optimizer_step = 0
    for epoch in range(config.training.epochs):
        epoch_loss = 0.0
        step_count = 0
        optimizer_step_count = 0
        total_batches = math.ceil(len(pairs) / config.training.batch_size)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(batched(pairs, config.training.batch_size), start=1):
            question_texts = [item[1] for item in batch]
            document_texts = [item[3] for item in batch]
            encoded_q = tokenizer(
                question_texts,
                padding=True,
                truncation=True,
                max_length=config.training.max_length,
                return_tensors="pt",
            )
            encoded_d = tokenizer(
                document_texts,
                padding=True,
                truncation=True,
                max_length=config.training.max_length,
                return_tensors="pt",
            )
            encoded_q = {key: value.to(device) for key, value in encoded_q.items()}
            encoded_d = {key: value.to(device) for key, value in encoded_d.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                query_outputs = model(**encoded_q)
                doc_outputs = model(**encoded_d)
                q_emb = _pool_embeddings(
                    query_outputs.last_hidden_state,
                    encoded_q["attention_mask"],
                    config.models.retriever.pooling,
                )
                d_emb = _pool_embeddings(
                    doc_outputs.last_hidden_state,
                    encoded_d["attention_mask"],
                    config.models.retriever.pooling,
                )
                if config.models.retriever.normalize:
                    q_emb = _normalize_matrix(q_emb)
                    d_emb = _normalize_matrix(d_emb)
                logits = q_emb @ d_emb.T / config.training.temperature
                labels = torch.arange(logits.shape[0], device=device)
                raw_loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = raw_loss / accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            raw_loss_value = float(raw_loss.detach().cpu().item())
            global_batch_step += 1
            did_optimizer_step = 0
            if _should_step_optimizer(batch_index, total_batches, accumulation_steps):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1
                global_optimizer_step += 1
                did_optimizer_step = 1
            step_history.append(
                {
                    "step": global_batch_step,
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "loss": raw_loss_value,
                    "optimizer_step": global_optimizer_step,
                    "did_optimizer_step": did_optimizer_step,
                }
            )
            epoch_loss += raw_loss_value
            step_count += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / max(step_count, 1),
                "optimizer_steps": optimizer_step_count,
                "gradient_accumulation_steps": accumulation_steps,
            }
        )
        LOGGER.info(
            "Retriever epoch %s/%s loss=%.4f",
            epoch + 1,
            config.training.epochs,
            history[-1]["loss"],
        )
    model.save_pretrained(run_dir / "retriever")
    tokenizer.save_pretrained(run_dir / "retriever")
    write_json(
        run_dir / "training_metrics.json",
        {
            "history": history,
            "step_history": step_history,
            "pair_count": len(pairs),
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "gradient_accumulation_steps": accumulation_steps,
        },
    )
    return run_dir / "retriever"
