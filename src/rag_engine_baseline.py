from __future__ import annotations

import time

from config import BASELINE_TOP_K, BASE_RETRIEVER_MODEL
from llm_utils import generate_answer, load_generator
from retrieval import BioRetriever


class BaselineRAG:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.retriever = BioRetriever(BASE_RETRIEVER_MODEL, device=device)
        self.tokenizer, self.model = load_generator(device=device)

    def answer_question(self, question: str) -> dict[str, object]:
        start = time.perf_counter()
        docs = self.retriever.search(question, top_k=BASELINE_TOP_K)
        retrieval_latency = time.perf_counter() - start
        generation_start = time.perf_counter()
        answer = generate_answer(question, docs, self.tokenizer, self.model, device=self.device)
        generation_latency = time.perf_counter() - generation_start
        return {
            "answer": answer,
            "documents": docs,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "latency": retrieval_latency + generation_latency,
        }

    def generate_answer(self, question: str) -> str:
        return str(self.answer_question(question)["answer"])


if __name__ == "__main__":
    rag = BaselineRAG()
    question = "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
    print(rag.generate_answer(question))
