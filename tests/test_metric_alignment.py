from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from biorag.config import ProjectConfig
from biorag.evaluation import _macro_f1
from biorag.io import load_jsonl
from biorag.reranking import rerank_contexts
from biorag.types import QuestionRecord, RetrievedContext, ScoredDocument


def make_alignment_config(tmpdir: str) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "experiment": {"name": "alignment", "mode": "full_biorag"},
            "dataset": {"name": "toy"},
            "models": {
                "retriever": {
                    "backend": "lexical",
                    "model_name": "toy-retriever",
                },
                "reranker": {
                    "backend": "cross_encoder",
                    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                },
                "generator": {
                    "backend": "rule_based",
                    "model_name": "toy-generator",
                },
            },
            "training": {
                "enabled": False,
                "epochs": 0,
                "learning_rate": 2e-5,
                "batch_size": 4,
                "max_length": 128,
                "temperature": 0.05,
                "rerank_batch_size": 7,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "no",
                "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
            },
            "inference": {
                "retrieve_top_k": 20,
                "rerank_enabled": True,
                "rerank_top_k": 2,
                "final_top_k": 2,
                "abstain_when_empty": True,
                "abstain_threshold": 0.05,
                "max_prompt_characters": 500,
                "greedy_decoding": True,
            },
            "evaluation": {
                "compute_latency": True,
                "compute_rouge_l": True,
                "compute_bert_score": False,
                "summary_table_name": "results_summary.md",
                "report_name": "evaluation_report.json",
            },
            "input_dir": str(Path(tmpdir) / "inputs"),
            "output_dir": str(Path(tmpdir) / "outputs"),
            "run_name": "alignment-run",
            "device": "cpu",
        }
    )


class MetricAlignmentTests(unittest.TestCase):
    def test_yesno_macro_f1_uses_yes_and_no_labels_only(self) -> None:
        value = _macro_f1(
            ["yes", "no"],
            ["yes", "i dont know"],
            labels=["yes", "no"],
        )
        self.assertAlmostEqual(value, 0.5)

    def test_cross_encoder_rerank_uses_configured_batch_size(self) -> None:
        class FakeCrossEncoder:
            last_instance: "FakeCrossEncoder | None" = None

            def __init__(self, model_source: str, device: str) -> None:
                self.model_source = model_source
                self.device = device
                self.calls: list[dict[str, object]] = []
                FakeCrossEncoder.last_instance = self

            def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
                self.calls.append({"pairs": pairs, "batch_size": batch_size})
                return [float(index) for index, _ in enumerate(pairs, start=1)]

        fake_module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_alignment_config(tmpdir)
            questions = [
                QuestionRecord(
                    id="q1",
                    type="factoid",
                    body="Which marker is relevant?",
                )
            ]
            contexts = [
                RetrievedContext(
                    question_id="q1",
                    question_type="factoid",
                    question="Which marker is relevant?",
                    stage="retrieval",
                    candidates=[
                        ScoredDocument(
                            document_id="1",
                            score=0.1,
                            rank=1,
                            title="Doc 1",
                            text="lower score candidate",
                            metadata={"retrieval_score": 0.1},
                        ),
                        ScoredDocument(
                            document_id="2",
                            score=0.2,
                            rank=2,
                            title="Doc 2",
                            text="higher score candidate",
                            metadata={"retrieval_score": 0.2},
                        ),
                    ],
                )
            ]
            output_path = Path(tmpdir) / "reranked.jsonl"
            with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
                rerank_contexts(
                    questions,
                    contexts,
                    config,
                    output_path,
                    device="cpu",
                )
            self.assertTrue(output_path.exists())
            self.assertEqual(
                FakeCrossEncoder.last_instance.calls[0]["batch_size"],
                config.training.rerank_batch_size,
            )
            rows = load_jsonl(output_path)
            self.assertEqual(rows[0]["candidates"][0]["document_id"], "2")
            self.assertEqual(rows[0]["candidates"][0]["score"], 2.0)
            self.assertEqual(rows[0]["candidates"][0]["metadata"]["retrieval_score"], 0.2)
            self.assertEqual(rows[0]["candidates"][0]["metadata"]["rerank_score"], 2.0)


if __name__ == "__main__":
    unittest.main()
