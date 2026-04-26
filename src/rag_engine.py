from __future__ import annotations

import time

from config import FINAL_TOP_K, FINETUNED_RETRIEVER_DIR, RERANK_CANDIDATE_K
from llm_utils import generate_answer, load_generator
from reranker import BioReranker
from retrieval import BioRetriever


class BioRAG:
    def __init__(self, device: str = "cuda") -> None:
        if not FINETUNED_RETRIEVER_DIR.exists():
            raise FileNotFoundError(
                f"Missing finetuned retriever: {FINETUNED_RETRIEVER_DIR}. "
                "Run train_retriever.py and build_index.py --model finetuned first."
            )
        self.device = device
        self.retriever = BioRetriever(str(FINETUNED_RETRIEVER_DIR), device=device)
        self.reranker = BioReranker(device=device)
        self.tokenizer, self.model = load_generator(device=device)

    def answer_question(self, question: str) -> dict[str, object]:
        start = time.perf_counter()
        candidates = self.retriever.search(question, top_k=RERANK_CANDIDATE_K)
        retrieval_latency = time.perf_counter() - start
        rerank_start = time.perf_counter()
        docs = self.reranker.rerank(question, candidates, top_k=FINAL_TOP_K)
        rerank_latency = time.perf_counter() - rerank_start
        generation_start = time.perf_counter()
        answer = generate_answer(question, docs, self.tokenizer, self.model, device=self.device)
        generation_latency = time.perf_counter() - generation_start
        return {
            "answer": answer,
            "documents": docs,
            "retrieval_latency": retrieval_latency,
            "rerank_latency": rerank_latency,
            "generation_latency": generation_latency,
            "latency": retrieval_latency + rerank_latency + generation_latency,
        }

    def generate_answer(self, question: str) -> str:
        return str(self.answer_question(question)["answer"])


if __name__ == "__main__":
    rag = BioRAG()
    question = "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
    print(rag.generate_answer(question))
