from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TripleRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    subject: str = ""
    predicate: str = ""
    object: str = ""


class SnippetRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    document: str = ""
    begin_section: str = ""
    end_section: str = ""
    offset_in_begin_section: int = 0
    offset_in_end_section: int = 0


class QuestionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    body: str
    documents: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    exact_answer: list[list[str]] = Field(default_factory=list)
    ideal_answer: list[str] = Field(default_factory=list)
    triples: list[TripleRecord] = Field(default_factory=list)
    snippets: list[SnippetRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str = ""
    abstract: str = ""
    text: str
    source: str = "pubmed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str
    score: float
    rank: int
    title: str = ""
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_id: str
    question_type: str
    question: str
    stage: str
    candidates: list[ScoredDocument] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_id: str
    question_type: str
    answer: str
    ranked_answers: list[str] = Field(default_factory=list)
    retrieved_documents: list[str] = Field(default_factory=list)
    latency_seconds: float | None = None
    abstained: bool = False
    prompt_excerpt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    metrics: dict[str, float | None] = Field(default_factory=dict)
    per_type_metrics: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
