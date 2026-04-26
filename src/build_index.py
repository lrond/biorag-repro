from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    ALL_PROCESSED,
    BASE_RETRIEVER_MODEL,
    FINETUNED_RETRIEVER_DIR,
    INDEX_DIR,
    ensure_dirs,
)
from data_utils import load_jsonl


def unique_documents(processed_path: Path) -> list[dict[str, str]]:
    seen: set[str] = set()
    documents: list[dict[str, str]] = []
    for row in load_jsonl(processed_path):
        for document in row.get("document_contents", []) or []:
            pmid = str(document.get("pmid", "")).strip()
            text = str(document.get("text", "")).strip()
            if not pmid or not text or pmid in seen:
                continue
            seen.add(pmid)
            documents.append(
                {
                    "pmid": pmid,
                    "title": str(document.get("title", "")),
                    "text": text,
                }
            )
    if not documents:
        raise ValueError(f"No documents found in {processed_path}; run prepare_data.py first.")
    return documents


def resolve_model(model_arg: str) -> str:
    if model_arg == "base":
        return BASE_RETRIEVER_MODEL
    if model_arg == "finetuned":
        if not FINETUNED_RETRIEVER_DIR.exists():
            raise FileNotFoundError(
                f"Finetuned model not found: {FINETUNED_RETRIEVER_DIR}. "
                "Run train_retriever.py first."
            )
        return str(FINETUNED_RETRIEVER_DIR)
    return model_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes.")
    parser.add_argument("--input", type=Path, default=ALL_PROCESSED)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", default="base", help="base, finetuned, or model path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()
    args.index_dir.mkdir(parents=True, exist_ok=True)
    model_name = resolve_model(args.model)
    documents = unique_documents(args.input)
    texts = [document["text"] for document in documents]
    pmids = [document["pmid"] for document in documents]

    print(f"Loaded {len(documents)} unique documents")
    tokenized = [text.lower().split() for text in tqdm(texts, desc="BM25 tokenizing")]
    bm25 = BM25Okapi(tokenized)
    with (args.index_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(bm25, handle)

    print(f"Encoding documents with {model_name}")
    model = SentenceTransformer(model_name, device=args.device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(args.index_dir / "faiss.index"))
    np.save(args.index_dir / "embeddings.npy", embeddings)

    with (args.index_dir / "doc_ids.pkl").open("wb") as handle:
        pickle.dump(pmids, handle)
    with (args.index_dir / "doc_texts.pkl").open("wb") as handle:
        pickle.dump(texts, handle)
    with (args.index_dir / "documents.pkl").open("wb") as handle:
        pickle.dump(documents, handle)

    metadata = {
        "model": model_name,
        "document_count": len(documents),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_metric": "inner_product_on_normalized_embeddings",
    }
    (args.index_dir / "index_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
