from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import BASE_RETRIEVER_MODEL, INDEX_DIR, USE_BM25


class BioRetriever:
    def __init__(
        self,
        model_name: str = BASE_RETRIEVER_MODEL,
        index_dir: Path = INDEX_DIR,
        device: str = "cuda",
        use_bm25: bool = USE_BM25,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.use_bm25 = use_bm25

        with (self.index_dir / "doc_ids.pkl").open("rb") as handle:
            self.doc_ids: list[str] = pickle.load(handle)
        with (self.index_dir / "doc_texts.pkl").open("rb") as handle:
            self.doc_texts: list[str] = pickle.load(handle)
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.model = SentenceTransformer(model_name, device=device)

        self.bm25 = None
        bm25_path = self.index_dir / "bm25.pkl"
        if self.use_bm25 and bm25_path.exists():
            with bm25_path.open("rb") as handle:
                self.bm25 = pickle.load(handle)

    def _dense_search(self, query: str, top_k: int) -> list[dict[str, object]]:
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)
        results: list[dict[str, object]] = []
        for score, index in zip(scores[0], indices[0]):
            if int(index) < 0:
                continue
            results.append(
                {
                    "pmid": self.doc_ids[int(index)],
                    "text": self.doc_texts[int(index)],
                    "score": float(score),
                    "source": "dense",
                }
            )
        return results

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, object]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "pmid": self.doc_ids[int(index)],
                "text": self.doc_texts[int(index)],
                "score": float(scores[int(index)]),
                "source": "bm25",
            }
            for index in top_indices
        ]

    def search(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        dense_results = self._dense_search(query, top_k)
        if not self.use_bm25:
            return dense_results

        merged: list[dict[str, object]] = []
        seen: set[str] = set()
        for result in dense_results + self._bm25_search(query, top_k):
            pmid = str(result["pmid"])
            if pmid in seen:
                continue
            seen.add(pmid)
            merged.append(result)
            if len(merged) >= top_k:
                break
        return merged
