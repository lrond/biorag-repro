from __future__ import annotations

from biorag.config import ProjectConfig
from biorag.types import QuestionRecord, RetrievedContext


def build_context_block(context: RetrievedContext, max_characters: int) -> str:
    chunks: list[str] = []
    total = 0
    for candidate in context.candidates:
        snippet = (
            f"[Doc {candidate.rank}] PMID={candidate.document_id}\n"
            f"Title: {candidate.title}\n"
            f"{candidate.text}\n"
        )
        if total + len(snippet) > max_characters:
            break
        chunks.append(snippet)
        total += len(snippet)
    return "\n".join(chunks).strip()


def build_prompt(
    question: QuestionRecord,
    context: RetrievedContext,
    config: ProjectConfig,
) -> str:
    evidence = build_context_block(context, config.inference.max_prompt_characters)
    instruction = {
        "yesno": "Answer with Yes, No, or I don't know. Use the evidence only.",
        "factoid": "Give the best short biomedical answer. If unsupported, say I don't know.",
        "list": (
            "Return a concise list of answers, one item per line. If unsupported, say I don't know."
        ),
        "summary": (
            "Write a concise grounded summary. If the evidence is insufficient, say I don't know."
        ),
    }.get(question.type, "Answer the biomedical question using only the evidence.")
    return (
        "You are a biomedical question answering assistant.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Question Type: {question.type}\n"
        f"Question: {question.body}\n\n"
        f"Evidence:\n{evidence if evidence else '[No supporting evidence retrieved]'}\n\n"
        "Answer:"
    )
