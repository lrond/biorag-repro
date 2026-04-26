from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from biorag.retrieval import (
    _sample_epoch_training_pairs,
    _search_faiss_index,
    _should_step_optimizer,
)


class RetrievalTests(unittest.TestCase):
    def test_faiss_search_uses_persisted_index_when_available(self) -> None:
        class FakeQueries:
            dtype: str | None = None

            def astype(self, dtype: str) -> "FakeQueries":
                self.dtype = dtype
                return self

        class FakeIndex:
            def search(
                self,
                queries: FakeQueries,
                top_k: int,
            ) -> tuple[list[list[float]], list[list[int]]]:
                self.queries = queries
                self.top_k = top_k
                return [[0.9, 0.5]], [[3, 1]]

        class FakeFaiss:
            def __init__(self) -> None:
                self.index = FakeIndex()
                self.path = ""

            def read_index(self, path: str) -> FakeIndex:
                self.path = path
                return self.index

        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            (index_dir / "faiss.index").write_text("fake", encoding="utf-8")
            queries = FakeQueries()
            fake_faiss = FakeFaiss()

            with patch("biorag.retrieval._faiss", return_value=fake_faiss):
                result = _search_faiss_index(index_dir, queries, top_k=2)

            self.assertEqual(result, ([[0.9, 0.5]], [[3, 1]]))
            self.assertEqual(fake_faiss.path, str(index_dir / "faiss.index"))
            self.assertEqual(queries.dtype, "float32")
            self.assertEqual(fake_faiss.index.top_k, 2)

    def test_gradient_accumulation_steps_at_interval_and_epoch_end(self) -> None:
        self.assertFalse(_should_step_optimizer(1, total_batches=5, accumulation_steps=2))
        self.assertTrue(_should_step_optimizer(2, total_batches=5, accumulation_steps=2))
        self.assertTrue(_should_step_optimizer(5, total_batches=5, accumulation_steps=2))

    def test_single_positive_sampling_keeps_one_pair_per_question(self) -> None:
        pairs = [
            ("q1", "question 1", "d1", "doc 1"),
            ("q1", "question 1", "d2", "doc 2"),
            ("q2", "question 2", "d3", "doc 3"),
        ]

        sampled = _sample_epoch_training_pairs(
            pairs,
            positive_sampling="single_positive_per_question",
            seed=42,
            epoch=0,
        )

        self.assertEqual(len(sampled), 2)
        self.assertEqual({pair[0] for pair in sampled}, {"q1", "q2"})

    def test_all_positive_sampling_keeps_all_pairs(self) -> None:
        pairs = [
            ("q1", "question 1", "d1", "doc 1"),
            ("q1", "question 1", "d2", "doc 2"),
            ("q2", "question 2", "d3", "doc 3"),
        ]

        sampled = _sample_epoch_training_pairs(
            pairs,
            positive_sampling="all_positives",
            seed=42,
            epoch=0,
        )

        self.assertEqual(len(sampled), 3)


if __name__ == "__main__":
    unittest.main()
