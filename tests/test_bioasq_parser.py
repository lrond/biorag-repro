from __future__ import annotations

import unittest
from pathlib import Path

from biorag.bioasq import parse_bioasq_questions

FIXTURE = Path(__file__).parent / "fixtures" / "toy_bioasq.json"


class BioASQParserTests(unittest.TestCase):
    def test_parser_normalizes_question_types(self) -> None:
        questions = parse_bioasq_questions(FIXTURE)
        self.assertEqual(len(questions), 4)
        self.assertEqual(questions[0].type, "yesno")
        self.assertEqual(questions[1].type, "factoid")
        self.assertEqual(questions[2].type, "list")
        self.assertEqual(questions[3].type, "summary")

    def test_exact_answer_normalization(self) -> None:
        questions = parse_bioasq_questions(FIXTURE)
        self.assertEqual(questions[0].exact_answer, [["yes"]])
        self.assertEqual(questions[1].exact_answer, [["aspirin", "acetylsalicylic acid"]])
        self.assertEqual(questions[2].exact_answer, [["vitamin C"], ["vitamin E"]])
        self.assertEqual(questions[3].exact_answer, [])


if __name__ == "__main__":
    unittest.main()
