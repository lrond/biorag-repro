from __future__ import annotations

import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from config import (
    HOLDOUT_SIZE,
    NCBI_EMAIL,
    PUBMED_BATCH_SIZE,
    PUBMED_CACHE_DIR,
    PUBMED_SLEEP_SECONDS,
    RANDOM_SEED,
    TRAINING_MEMBER,
    TRAINING_ZIP,
    ensure_dirs,
)


def extract_pmid(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"pubmed/?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{5,})\b", text)
    return match.group(1) if match else ""


def read_bioasq_training(zip_path: Path = TRAINING_ZIP) -> list[dict[str, Any]]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing BioASQ archive: {zip_path}")
    with ZipFile(zip_path) as archive:
        names = archive.namelist()
        member = TRAINING_MEMBER if TRAINING_MEMBER in names else next(
            name for name in names if name.endswith(".json")
        )
        payload = json.loads(archive.read(member).decode("utf-8"))
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in {zip_path}")
    return questions


def stratified_holdout(
    questions: list[dict[str, Any]],
    sample_size: int = HOLDOUT_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sample_size <= 0 or sample_size >= len(questions):
        return [], list(questions)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        groups[str(question.get("type", "")).lower()].append(question)

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    total = len(questions)
    for qtype, items in groups.items():
        exact = sample_size * len(items) / total
        allocations[qtype] = int(exact)
        remainders.append((exact - int(exact), qtype))

    assigned = sum(allocations.values())
    for _, qtype in sorted(remainders, reverse=True):
        if assigned >= sample_size:
            break
        allocations[qtype] += 1
        assigned += 1

    rng = random.Random(seed)
    eval_questions: list[dict[str, Any]] = []
    for qtype, items in groups.items():
        bucket = list(items)
        rng.shuffle(bucket)
        eval_questions.extend(bucket[: allocations[qtype]])
    rng.shuffle(eval_questions)

    eval_ids = {str(question.get("id", "")) for question in eval_questions}
    train_questions = [
        question for question in questions if str(question.get("id", "")) not in eval_ids
    ]
    return train_questions, eval_questions


def question_pmids(question: dict[str, Any]) -> list[str]:
    pmids: list[str] = []
    for document in question.get("documents", []) or []:
        pmid = extract_pmid(document)
        if pmid:
            pmids.append(pmid)
    for snippet in question.get("snippets", []) or []:
        pmid = extract_pmid(snippet.get("document", ""))
        if pmid:
            pmids.append(pmid)
    return sorted(set(pmids))


def all_pmids(questions: list[dict[str, Any]]) -> list[str]:
    pmids: set[str] = set()
    for question in questions:
        pmids.update(question_pmids(question))
    return sorted(pmids)


def _cache_path(pmid: str) -> Path:
    return PUBMED_CACHE_DIR / f"{pmid}.json"


def _load_cached(pmid: str) -> dict[str, str] | None:
    path = _cache_path(pmid)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cached(document: dict[str, str]) -> None:
    _cache_path(document["pmid"]).write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_pubmed_batch(pmids: list[str]) -> list[dict[str, str]]:
    query = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": "biorag",
    }
    if NCBI_EMAIL:
        query["email"] = NCBI_EMAIL
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    url = f"{url}?{urllib.parse.urlencode(query)}"
    print(f"Fetching PubMed batch with {len(pmids)} PMIDs")
    with urllib.request.urlopen(url, timeout=60) as response:
        raw_xml = response.read()
    root = ET.fromstring(raw_xml)
    documents: list[dict[str, str]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="").strip()
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            if text:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts).strip()
        text = "\n".join(part for part in [title, abstract] if part).strip()
        if pmid and text:
            documents.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "text": text,
                }
            )
    return documents


def load_or_fetch_pubmed(pmids: list[str], offline: bool = False) -> dict[str, dict[str, str]]:
    ensure_dirs()
    documents: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for pmid in pmids:
        cached = _load_cached(pmid)
        if cached:
            documents[pmid] = cached
        else:
            missing.append(pmid)

    if offline:
        return documents

    for start in range(0, len(missing), PUBMED_BATCH_SIZE):
        batch = missing[start : start + PUBMED_BATCH_SIZE]
        try:
            fetched = fetch_pubmed_batch(batch)
        except Exception as error:
            print(f"WARNING: PubMed fetch failed for batch starting {batch[0]}: {error}")
            continue
        for document in fetched:
            documents[document["pmid"]] = document
            _write_cached(document)
        print(f"Cached {len(documents)}/{len(pmids)} PubMed documents")
        time.sleep(PUBMED_SLEEP_SECONDS)
    return documents


def snippet_fallbacks(question: dict[str, Any]) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for snippet in question.get("snippets", []) or []:
        pmid = extract_pmid(snippet.get("document", ""))
        text = str(snippet.get("text", "")).strip()
        if pmid and text:
            grouped[pmid].append(text)
    return {
        pmid: {
            "pmid": pmid,
            "title": "",
            "abstract": " ".join(parts),
            "text": " ".join(parts),
        }
        for pmid, parts in grouped.items()
    }


def build_processed_record(
    question: dict[str, Any],
    documents: dict[str, dict[str, str]],
) -> dict[str, Any]:
    fallbacks = snippet_fallbacks(question)
    document_contents = []
    for pmid in question_pmids(question):
        document = documents.get(pmid) or fallbacks.get(pmid)
        if document and document.get("text"):
            document_contents.append(document)
    return {
        "id": str(question.get("id", "")).strip(),
        "type": str(question.get("type", "")).strip().lower(),
        "body": str(question.get("body", "")).strip(),
        "documents": question.get("documents", []) or [],
        "snippets": question.get("snippets", []) or [],
        "exact_answer": question.get("exact_answer", []),
        "ideal_answer": question.get("ideal_answer", []),
        "document_contents": document_contents,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"WARNING: skipped malformed JSONL line {line_number} in {path}: {error}")
    return rows


def type_counts(questions: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(question.get("type", "")).lower() for question in questions))
