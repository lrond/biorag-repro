from __future__ import annotations

import math
from collections import defaultdict

from biorag.types import QuestionRecord
from biorag.utils import seeded_random


def stratified_sample_questions(
    questions: list[QuestionRecord],
    sample_size: int,
    seed: int,
) -> list[QuestionRecord]:
    if sample_size <= 0 or sample_size >= len(questions):
        return list(questions)
    groups: dict[str, list[QuestionRecord]] = defaultdict(list)
    for question in questions:
        groups[question.type].append(question)
    total = len(questions)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for question_type, items in groups.items():
        exact = sample_size * len(items) / total
        allocation = min(len(items), math.floor(exact))
        allocations[question_type] = allocation
        remainders.append((exact - allocation, question_type))
    assigned = sum(allocations.values())
    for _, question_type in sorted(remainders, reverse=True):
        if assigned >= sample_size:
            break
        if allocations[question_type] < len(groups[question_type]):
            allocations[question_type] += 1
            assigned += 1
    rng = seeded_random(seed)
    sampled: list[QuestionRecord] = []
    for question_type, items in groups.items():
        bucket = list(items)
        rng.shuffle(bucket)
        sampled.extend(bucket[: allocations[question_type]])
    rng.shuffle(sampled)
    return sampled


def select_questions_by_ids(
    questions: list[QuestionRecord],
    selected_ids: set[str],
) -> list[QuestionRecord]:
    return [question for question in questions if question.id in selected_ids]
