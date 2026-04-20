from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from biorag.bioasq import parse_bioasq_questions
from biorag.config import ProjectConfig
from biorag.datasets import materialize_question_splits
from biorag.io import load_jsonl, read_json

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _write_zip(zip_path: Path, member_name: str, payload: dict) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(member_name, json.dumps(payload))


class DatasetProtocolTests(unittest.TestCase):
    def test_zip_parser_normalizes_pubmed_urls_to_pmids(self) -> None:
        payload = {
            "questions": [
                {
                    "id": "zip-q1",
                    "type": "factoid",
                    "body": "Which drug was used?",
                    "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/123456"],
                    "snippets": [
                        {
                            "text": "Drug X was used in the trial.",
                            "document": "https://pubmed.ncbi.nlm.nih.gov/987654/",
                            "beginSection": "abstract",
                            "endSection": "abstract",
                            "offsetInBeginSection": 0,
                            "offsetInEndSection": 10,
                        }
                    ],
                    "exact_answer": ["Drug X"],
                    "ideal_answer": ["Drug X was used in the trial."],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "toy.zip"
            _write_zip(archive_path, "nested/toy.json", payload)
            questions = parse_bioasq_questions(archive_path, member_names=["nested/toy.json"])
            self.assertEqual(questions[0].documents, ["123456"])
            self.assertEqual(questions[0].snippets[0].document, "987654")

    def test_report_holdout_protocol_keeps_train_eval_disjoint(self) -> None:
        toy_payload = json.loads((FIXTURE_DIR / "toy_bioasq.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "toy.zip"
            _write_zip(archive_path, "BioASQ-training12b/training12b_new.json", toy_payload)
            config = ProjectConfig.model_validate(
                {
                    "experiment": {"name": "toy-report", "mode": "full_biorag"},
                    "dataset": {
                        "name": "toy-report",
                        "question_path": str(archive_path),
                        "protocol": "report_holdout",
                        "sample_size": 2,
                        "sample_seed": 42,
                        "corpus_mode": "pubmed_dump",
                        "cache_dir": str(Path(tmpdir) / "cache"),
                        "training": {
                            "path": str(archive_path),
                            "member_names": ["BioASQ-training12b/training12b_new.json"],
                        },
                        "pubmed_dump": {
                            "input_path": str(FIXTURE_DIR / "toy_pubmed_dump.jsonl"),
                            "id_field": "pmid",
                            "title_field": "title",
                            "abstract_field": "abstract",
                        },
                    },
                    "models": {
                        "retriever": {"backend": "lexical", "model_name": "toy-lexical"},
                        "reranker": {"backend": "lexical", "model_name": "toy-reranker"},
                        "generator": {"backend": "rule_based", "model_name": "toy-generator"},
                    },
                    "training": {"enabled": False},
                }
            )
            artifacts = materialize_question_splits(config, Path(tmpdir) / "splits")
            train_ids = {row["id"] for row in load_jsonl(artifacts["train"])}
            evaluation_ids = {row["id"] for row in load_jsonl(artifacts["evaluation"])}
            manifest = read_json(artifacts["manifest"])
            self.assertEqual(len(train_ids), 2)
            self.assertEqual(len(evaluation_ids), 2)
            self.assertTrue(train_ids.isdisjoint(evaluation_ids))
            self.assertEqual(manifest["protocol"], "report_holdout")
            self.assertEqual(manifest["evaluation_count"], 2)


if __name__ == "__main__":
    unittest.main()
