from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from biorag.bioasq import build_training_pairs
from biorag.config import ProjectConfig, resolve_model_source
from biorag.io import dump_jsonl, load_jsonl, write_json
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


def _lexical_score(query: str, document: str) -> float:
    query_tokens = set(simple_tokenize(query))
    document_tokens = set(simple_tokenize(document))
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(document_tokens))


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
        write_json(index_dir / "index_metadata.json", {"backend": backend, "document_count": len(documents)})
        return index_dir
    if backend != "transformer":
        raise ValueError(f"Unsupported retriever backend: {backend}")
    np = _numpy()
    torch, AutoModel, AutoTokenizer = _transformers()
    source = resolve_model_source(
        config.models.retriever.model_name,
        checkpoint_path=config.models.retriever.checkpoint_path,
        override_path=model_source,
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)
    model.to(device)
    model.eval()
    embeddings = []
    doc_ids = []
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
            outputs = model(**encoded)
            pooled = _pool_embeddings(outputs.last_hidden_state, encoded["attention_mask"], config.models.retriever.pooling)
            if config.models.retriever.normalize:
                pooled = _normalize_matrix(pooled)
            embeddings.append(pooled.cpu().numpy())
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


def _load_transformer_assets(index_dir: Path) -> tuple[list[DocumentRecord], list[str], Any]:
    np = _numpy()
    documents = [DocumentRecord.model_validate(row) for row in load_jsonl(index_dir / "documents.jsonl")]
    doc_ids = load_jsonl(index_dir / "documents.jsonl")
    ids = [row["id"] for row in doc_ids]
    embeddings = np.load(index_dir / "embeddings.npy")
    return documents, ids, embeddings


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
                    candidates=[
                        ScoredDocument(
                            document_id=document_id,
                            score=float(score),
                            rank=rank + 1,
                            title=documents_by_id[document_id].title,
                            text=documents_by_id[document_id].text,
                        )
                        for rank, (document_id, score) in enumerate(ranked)
                    ],
                )
            )
        return dump_jsonl(output_path, contexts)
    if backend != "transformer":
        raise ValueError(f"Unsupported retriever backend: {backend}")
    np = _numpy()
    torch, AutoModel, AutoTokenizer = _transformers()
    index_path = Path(index_dir)
    docs, doc_ids, doc_embeddings = _load_transformer_assets(index_path)
    documents_by_id = {document.id: document for document in docs}
    source = resolve_model_source(
        config.models.retriever.model_name,
        checkpoint_path=config.models.retriever.checkpoint_path,
        override_path=model_source,
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)
    model.to(device)
    model.eval()
    question_embeddings = []
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
            outputs = model(**encoded)
            pooled = _pool_embeddings(outputs.last_hidden_state, encoded["attention_mask"], config.models.retriever.pooling)
            if config.models.retriever.normalize:
                pooled = _normalize_matrix(pooled)
            question_embeddings.append(pooled.cpu().numpy())
    queries = np.concatenate(question_embeddings, axis=0)
    scores = queries @ doc_embeddings.T
    for row_index, question in enumerate(questions):
        ranking = scores[row_index].argsort()[::-1][: config.inference.retrieve_top_k]
        candidates = []
        for rank, doc_index in enumerate(ranking):
            document_id = doc_ids[int(doc_index)]
            document = documents_by_id[document_id]
            candidates.append(
                ScoredDocument(
                    document_id=document_id,
                    score=float(scores[row_index][int(doc_index)]),
                    rank=rank + 1,
                    title=document.title,
                    text=document.text,
                )
            )
        contexts.append(
            RetrievedContext(
                question_id=question.id,
                question_type=question.type,
                question=question.body,
                stage="retrieval",
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
    model = AutoModel.from_pretrained(config.models.retriever.model_name)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(config.training.epochs):
        epoch_loss = 0.0
        step_count = 0
        for batch in batched(pairs, config.training.batch_size):
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
            query_outputs = model(**encoded_q)
            doc_outputs = model(**encoded_d)
            q_emb = _pool_embeddings(query_outputs.last_hidden_state, encoded_q["attention_mask"], config.models.retriever.pooling)
            d_emb = _pool_embeddings(doc_outputs.last_hidden_state, encoded_d["attention_mask"], config.models.retriever.pooling)
            if config.models.retriever.normalize:
                q_emb = _normalize_matrix(q_emb)
                d_emb = _normalize_matrix(d_emb)
            logits = q_emb @ d_emb.T / config.training.temperature
            labels = torch.arange(logits.shape[0], device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += float(loss.detach().cpu().item())
            step_count += 1
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(step_count, 1)})
        LOGGER.info("Retriever epoch %s/%s loss=%.4f", epoch + 1, config.training.epochs, history[-1]["loss"])
    model.save_pretrained(run_dir / "retriever")
    tokenizer.save_pretrained(run_dir / "retriever")
    write_json(run_dir / "training_metrics.json", {"history": history, "pair_count": len(pairs)})
    return run_dir / "retriever"
