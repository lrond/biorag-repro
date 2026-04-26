from __future__ import annotations

import tempfile
import unittest

from biorag.generation import _should_abstain
from biorag.types import QuestionRecord, RetrievedContext, ScoredDocument
from tests.test_metric_alignment import make_alignment_config


class GenerationTests(unittest.TestCase):
    def test_abstain_uses_retrieval_score_after_reranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_alignment_config(tmpdir)
            question = QuestionRecord(id="q1", type="factoid", body="Which marker?")
            context = RetrievedContext(
                question_id="q1",
                question_type="factoid",
                question="Which marker?",
                stage="reranked",
                metadata={"top_retrieval_score": 0.2},
                candidates=[
                    ScoredDocument(
                        document_id="1",
                        score=-2.0,
                        rank=1,
                        text="useful evidence",
                        metadata={"retrieval_score": 0.2, "rerank_score": -2.0},
                    )
                ],
            )
            self.assertFalse(_should_abstain(question, context, config))

    def test_abstain_still_rejects_low_retrieval_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_alignment_config(tmpdir)
            question = QuestionRecord(id="q1", type="factoid", body="Which marker?")
            context = RetrievedContext(
                question_id="q1",
                question_type="factoid",
                question="Which marker?",
                stage="reranked",
                metadata={"top_retrieval_score": 0.2},
                candidates=[
                    ScoredDocument(
                        document_id="1",
                        score=0.01,
                        rank=1,
                        text="weak evidence",
                        metadata={"retrieval_score": 0.01},
                    )
                ],
            )
            self.assertTrue(_should_abstain(question, context, config))

    def test_rerank_safety_gate_can_abstain_on_low_rerank_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_alignment_config(tmpdir)
            config = config.model_copy(
                update={
                    "inference": config.inference.model_copy(
                        update={"rerank_abstain_threshold": 0.0}
                    )
                }
            )
            question = QuestionRecord(id="q1", type="factoid", body="Which marker?")
            context = RetrievedContext(
                question_id="q1",
                question_type="factoid",
                question="Which marker?",
                stage="reranked",
                metadata={"top_retrieval_score": 0.8},
                candidates=[
                    ScoredDocument(
                        document_id="1",
                        score=-1.2,
                        rank=1,
                        text="weak reranked evidence",
                        metadata={"retrieval_score": 0.8, "rerank_score": -1.2},
                    )
                ],
            )
            self.assertTrue(_should_abstain(question, context, config))


if __name__ == "__main__":
    unittest.main()
