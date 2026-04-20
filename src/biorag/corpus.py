from __future__ import annotations

import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from biorag.config import DatasetConfig
from biorag.io import dump_jsonl, load_jsonl, write_json
from biorag.types import DocumentRecord, QuestionRecord
from biorag.utils import chunked_iterable, ensure_dir, get_logger

LOGGER = get_logger(__name__)


def load_corpus(path: str | Path) -> list[DocumentRecord]:
    return [DocumentRecord.model_validate(row) for row in load_jsonl(path)]


def build_corpus_lookup(documents: list[DocumentRecord]) -> dict[str, DocumentRecord]:
    return {document.id: document for document in documents}


def extract_linked_pmids(questions: list[QuestionRecord]) -> list[str]:
    pmids: set[str] = set()
    for question in questions:
        pmids.update(document_id for document_id in question.documents if document_id)
        pmids.update(snippet.document for snippet in question.snippets if snippet.document)
    return sorted(pmids)


class PubMedClient:
    def __init__(self, dataset_config: DatasetConfig) -> None:
        self.base_url = dataset_config.linked_pubmed.base_url
        self.tool = dataset_config.linked_pubmed.tool
        self.email = os.getenv(dataset_config.linked_pubmed.email_env, "")
        self.batch_size = dataset_config.linked_pubmed.batch_size
        self.sleep_seconds = dataset_config.linked_pubmed.sleep_seconds

    def fetch_many(self, pmids: list[str]) -> list[DocumentRecord]:
        documents: list[DocumentRecord] = []
        for batch in chunked_iterable(pmids, self.batch_size):
            documents.extend(self._fetch_batch(batch))
            time.sleep(self.sleep_seconds)
        return documents

    def _fetch_batch(self, pmids: list[str]) -> list[DocumentRecord]:
        query = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": self.tool,
        }
        if self.email:
            query["email"] = self.email
        url = f"{self.base_url}/efetch.fcgi?{urllib.parse.urlencode(query)}"
        LOGGER.info("Fetching PubMed batch with %s PMIDs", len(pmids))
        with urllib.request.urlopen(url, timeout=60) as response:
            raw_xml = response.read()
        root = ET.fromstring(raw_xml)
        documents: list[DocumentRecord] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", default="").strip()
            title = "".join(article.find(".//ArticleTitle").itertext()).strip() if article.find(".//ArticleTitle") is not None else ""
            abstract_parts = []
            for node in article.findall(".//Abstract/AbstractText"):
                text = "".join(node.itertext()).strip()
                if text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts).strip()
            text = "\n\n".join(part for part in [title, abstract] if part).strip()
            if pmid and text:
                documents.append(
                    DocumentRecord(
                        id=pmid,
                        title=title,
                        abstract=abstract,
                        text=text,
                        source="linked_pubmed",
                    )
                )
        return documents


def build_linked_pubmed_corpus(
    questions: list[QuestionRecord],
    dataset_config: DatasetConfig,
    output_path: str | Path,
) -> tuple[Path, Path]:
    cache_dir = ensure_dir(dataset_config.cache_dir)
    pmids = extract_linked_pmids(questions)
    cached_documents: dict[str, DocumentRecord] = {}
    for pmid in pmids:
        cache_path = cache_dir / f"{pmid}.json"
        if cache_path.exists():
            cached_documents[pmid] = DocumentRecord.model_validate_json(cache_path.read_text(encoding="utf-8"))
    missing = [pmid for pmid in pmids if pmid not in cached_documents]
    if missing:
        client = PubMedClient(dataset_config)
        fetched = client.fetch_many(missing)
        for document in fetched:
            cached_documents[document.id] = document
            (cache_dir / f"{document.id}.json").write_text(
                document.model_dump_json(indent=2),
                encoding="utf-8",
            )
    ordered = [cached_documents[pmid] for pmid in pmids if pmid in cached_documents]
    corpus_path = dump_jsonl(output_path, ordered)
    manifest_path = write_json(
        Path(output_path).with_name("corpus_manifest.json"),
        {
            "mode": "linked_pubmed",
            "requested_pmids": len(pmids),
            "resolved_documents": len(ordered),
            "missing_pmids": [pmid for pmid in pmids if pmid not in cached_documents],
        },
    )
    return corpus_path, manifest_path


def build_pubmed_dump_corpus(
    questions: list[QuestionRecord],
    dataset_config: DatasetConfig,
    output_path: str | Path,
) -> tuple[Path, Path]:
    pmids = set(extract_linked_pmids(questions))
    raw_rows = load_jsonl(dataset_config.pubmed_dump.input_path)
    documents: list[DocumentRecord] = []
    for row in raw_rows:
        pmid = str(row.get(dataset_config.pubmed_dump.id_field, "")).strip()
        if pmid not in pmids:
            continue
        title = str(row.get(dataset_config.pubmed_dump.title_field, "")).strip()
        abstract = str(row.get(dataset_config.pubmed_dump.abstract_field, "")).strip()
        text = "\n\n".join(part for part in [title, abstract] if part).strip()
        if text:
            documents.append(
                DocumentRecord(
                    id=pmid,
                    title=title,
                    abstract=abstract,
                    text=text,
                    source="pubmed_dump",
                )
            )
    corpus_path = dump_jsonl(output_path, documents)
    manifest_path = write_json(
        Path(output_path).with_name("corpus_manifest.json"),
        {
            "mode": "pubmed_dump",
            "requested_pmids": len(pmids),
            "resolved_documents": len(documents),
            "missing_pmids": sorted(pmids - {document.id for document in documents}),
        },
    )
    return corpus_path, manifest_path
