from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from biorag.config import ProjectConfig
from biorag.pipeline import build_corpus_stage, build_index_stage, generate_stage, retrieve_stage


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def make_toy_config(tmpdir: str) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "experiment": {"name": "toy", "mode": "baseline"},
            "runtime": {"seed": 42, "num_workers": 0, "log_level": "INFO"},
            "paths": {
                "cache_dir": str(Path(tmpdir) / "cache"),
                "artifact_dir": str(Path(tmpdir) / "outputs"),
                "processed_dir": str(Path(tmpdir) / "processed")
            },
            "dataset": {
                "name": "toy",
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
            },
            "models": {
                "retriever": {
                    "backend": "lexical",
                    "model_name": "toy-lexical",
                    "pooling": "cls",
                    "normalize": True,
                    "freeze": True
                },
                "reranker": {
                    "backend": "lexical",
                    "model_name": "toy-reranker"
                },
                "generator": {
                    "backend": "rule_based",
                    "model_name": "toy-generator",
                    "max_new_tokens": 64,
                    "do_sample": False,
                    "temperature": 0.0
                }
            },
            "training": {
                "enabled": False,
                "epochs": 0,
                "learning_rate": 2e-5,
                "batch_size": 4,
                "max_length": 128,
                "temperature": 0.05,
                "rerank_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "no",
                "checkpoint_dir": str(Path(tmpdir) / "checkpoints")
            },
            "inference": {
                "retrieve_top_k": 3,
                "rerank_enabled": True,
                "rerank_top_k": 3,
                "final_top_k": 2,
                "abstain_when_empty": True,
                "abstain_threshold": 0.05,
                "max_prompt_characters": 2000,
                "greedy_decoding": True
            },
            "evaluation": {
                "compute_latency": True,
                "compute_rouge_l": True,
                "compute_bert_score": False,
                "summary_table_name": "results_summary.md",
                "report_name": "evaluation_report.json"
            },
            "input_dir": str(FIXTURE_DIR),
            "output_dir": str(Path(tmpdir) / "outputs"),
            "run_name": "toy-run",
            "device": "cpu"
        }
    )


class PipelineContractTests(unittest.TestCase):
    def test_stage_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_toy_config(tmpdir)
            corpus_paths = build_corpus_stage(config)
            index_paths = build_index_stage(config)
            retrieval_paths = retrieve_stage(config)
            generation_paths = generate_stage(config)
            self.assertTrue(corpus_paths["corpus"].exists())
            self.assertTrue(index_paths["index"].exists())
            self.assertTrue(retrieval_paths["retrieved"].exists())
            self.assertTrue(generation_paths["predictions"].exists())


if __name__ == "__main__":
    unittest.main()
