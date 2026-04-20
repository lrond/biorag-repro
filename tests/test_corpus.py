from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from biorag.bioasq import parse_bioasq_questions
from biorag.config import DatasetConfig
from biorag.corpus import build_pubmed_dump_corpus, load_corpus


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
                        "abstract_field": "abstract"
                    }
                }
            )
            corpus_path, manifest_path = build_pubmed_dump_corpus(
                questions,
                dataset_config,
                Path(tmpdir) / "corpus.jsonl",
            )
            corpus = load_corpus(corpus_path)
            self.assertEqual(len(corpus), 3)
            self.assertTrue(manifest_path.exists())
            self.assertIn("Aspirin and fever", corpus[0].text)


if __name__ == "__main__":
    unittest.main()
