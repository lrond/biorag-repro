from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from biorag.bioasq import parse_bioasq_questions
from biorag.config import DatasetConfig
from biorag.corpus import build_linked_pubmed_corpus, build_pubmed_dump_corpus, load_corpus
from biorag.types import DocumentRecord, QuestionRecord

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class CorpusTests(unittest.TestCase):
    def test_pubmed_dump_corpus_builds_expected_documents(self) -> None:
        questions = parse_bioasq_questions(FIXTURE_DIR / "toy_bioasq.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_config = DatasetConfig.model_validate(
                {
                    "name": "toy_pubmed_dump",
                    "question_path": str(FIXTURE_DIR / "toy_bioasq.json"),
                    "sample_size": 4,
                    "sample_seed": 42,
                    "corpus_mode": "pubmed_dump",
                    "cache_dir": str(Path(tmpdir) / "cache"),
                    "canonical_questions_path": str(Path(tmpdir) / "questions.jsonl"),
                    "sampled_questions_path": str(Path(tmpdir) / "sampled.json"),
                    "pubmed_dump": {
                        "input_path": str(FIXTURE_DIR / "toy_pubmed_dump.jsonl"),
                        "id_field": "pmid",
                        "title_field": "title",
                        "abstract_field": "abstract",
                    },
                }
            )
            with patch("biorag.corpus.load_jsonl", side_effect=AssertionError):
                corpus_path, manifest_path = build_pubmed_dump_corpus(
                    questions,
                    dataset_config,
                    Path(tmpdir) / "corpus.jsonl",
                )
            corpus = load_corpus(corpus_path)
            self.assertEqual(len(corpus), 3)
            self.assertTrue(manifest_path.exists())
            self.assertIn("Aspirin and fever", corpus[0].text)

    def test_linked_pubmed_corpus_caches_each_fetched_batch(self) -> None:
        questions = [
            QuestionRecord(
                id="q1",
                type="factoid",
                body="Which drug?",
                documents=["1", "2"],
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_config = DatasetConfig.model_validate(
                {
                    "name": "toy_linked_pubmed",
                    "corpus_mode": "linked_pubmed",
                    "cache_dir": str(Path(tmpdir) / "cache"),
                    "linked_pubmed": {
                        "base_url": "https://example.test",
                        "tool": "toy",
                        "batch_size": 1,
                        "sleep_seconds": 0,
                    },
                }
            )
            fetched_batches = iter(
                [
                    [
                        DocumentRecord(
                            id="1",
                            title="Doc 1",
                            text="Doc 1 text",
                            source="linked_pubmed",
                        )
                    ],
                    [
                        DocumentRecord(
                            id="2",
                            title="Doc 2",
                            text="Doc 2 text",
                            source="linked_pubmed",
                        )
                    ],
                ]
            )

            with patch(
                "biorag.corpus.PubMedClient.iter_fetch_batches",
                return_value=fetched_batches,
            ):
                corpus_path, manifest_path = build_linked_pubmed_corpus(
                    questions,
                    dataset_config,
                    Path(tmpdir) / "corpus.jsonl",
                )

            self.assertTrue((Path(tmpdir) / "cache" / "1.json").exists())
            self.assertTrue((Path(tmpdir) / "cache" / "2.json").exists())
            self.assertEqual(len(load_corpus(corpus_path)), 2)
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
