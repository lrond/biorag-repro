from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from config import RERANK_BATCH_SIZE, RERANKER_MODEL


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class BioReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, device: str = "cuda") -> None:
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        question: str,
        docs: list[dict[str, object]],
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        if not docs:
            return []
        pairs = [(question, str(doc["text"])) for doc in docs]
        logits = self.model.predict(pairs, batch_size=RERANK_BATCH_SIZE)
        rescored = []
        for doc, logit in zip(docs, logits):
            item = dict(doc)
            item["rerank_logit"] = float(logit)
            item["rerank_score"] = sigmoid(float(logit))
            rescored.append(item)
        rescored.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
        return rescored[:top_k]
