from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from biorag.types import QuestionRecord, SnippetRecord, TripleRecord


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def normalize_exact_answer(question_type: str, raw_value: Any) -> list[list[str]]:
    qtype = (question_type or "").lower()
    if qtype == "summary":
        return []
    if qtype == "yesno":
        values = _as_string_list(raw_value)
        normalized = values[0].lower() if values else "unknown"
        return [[normalized]]
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [[raw_value.strip()]]
    if isinstance(raw_value, list):
        if not raw_value:
            return []
        if all(isinstance(item, str) for item in raw_value):
            if qtype == "list":
                return [[item.strip()] for item in raw_value if item.strip()]
            return [[item.strip() for item in raw_value if item.strip()]]
        normalized_groups: list[list[str]] = []
        for item in raw_value:
            if isinstance(item, list):
                group = [str(choice).strip() for choice in item if str(choice).strip()]
                if group:
                    normalized_groups.append(group)
            elif isinstance(item, str) and item.strip():
                normalized_groups.append([item.strip()])
        return normalized_groups
    return [[str(raw_value).strip()]]


def _normalize_snippets(raw_snippets: list[dict[str, Any]] | None) -> list[SnippetRecord]:
    snippets: list[SnippetRecord] = []
    for snippet in raw_snippets or []:
        snippets.append(
            SnippetRecord(
                text=snippet.get("text", ""),
                document=snippet.get("document", ""),
                begin_section=snippet.get("beginSection", ""),
                end_section=snippet.get("endSection", ""),
                offset_in_begin_section=int(snippet.get("offsetInBeginSection", 0) or 0),
                offset_in_end_section=int(snippet.get("offsetInEndSection", 0) or 0),
            )
        )
    return snippets


def _normalize_triples(raw_triples: list[dict[str, Any]] | None) -> list[TripleRecord]:
    triples: list[TripleRecord] = []
    for triple in raw_triples or []:
        triples.append(
            TripleRecord(
                subject=triple.get("s", triple.get("subject", "")),
                predicate=triple.get("p", triple.get("predicate", "")),
                object=triple.get("o", triple.get("object", "")),
            )
        )
    return triples


def parse_bioasq_questions(path: str | Path) -> list[QuestionRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records: list[QuestionRecord] = []
    for raw in payload.get("questions", []):
        records.append(
            QuestionRecord(
                id=str(raw.get("id", "")).strip(),
                type=str(raw.get("type", "")).strip().lower(),
                body=str(raw.get("body", "")).strip(),
                documents=_as_string_list(raw.get("documents")),
                concepts=_as_string_list(raw.get("concepts")),
                exact_answer=normalize_exact_answer(raw.get("type", ""), raw.get("exact_answer")),
                ideal_answer=_as_string_list(raw.get("ideal_answer")),
                triples=_normalize_triples(raw.get("triples")),
                snippets=_normalize_snippets(raw.get("snippets")),
                metadata={"source_path": str(path)},
            )
        )
    return records


def build_training_pairs(
    questions: list[QuestionRecord],
    corpus_by_id: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    pairs: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for question in questions:
        candidate_doc_ids = list(question.documents)
        if not candidate_doc_ids:
            candidate_doc_ids = [snippet.document for snippet in question.snippets if snippet.document]
        for document_id in candidate_doc_ids:
            if document_id not in corpus_by_id:
                continue
            key = (question.id, document_id)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((question.id, question.body, document_id, corpus_by_id[document_id]))
    return pairs
