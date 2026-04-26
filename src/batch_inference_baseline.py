from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from config import BASELINE_PREDICTIONS, EVAL_PROCESSED, ensure_dirs
from data_utils import load_jsonl
from rag_engine_baseline import BaselineRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline batch inference.")
    parser.add_argument("--input", type=Path, default=EVAL_PROCESSED)
    parser.add_argument("--output", type=Path, default=BASELINE_PREDICTIONS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    ensure_dirs()
    questions = load_jsonl(args.input)
    if args.limit:
        questions = questions[: args.limit]
    rag = BaselineRAG(device=args.device)
    output = {"questions": []}

    for index, row in enumerate(tqdm(questions), start=1):
        result = rag.answer_question(row["body"])
        item = {
            "id": row["id"],
            "body": row["body"],
            "type": row["type"],
            "ideal_answer": result["answer"],
            "exact_answer": "",
            "documents": [doc["pmid"] for doc in result["documents"]],
            "metadata": {
                "latency": result["latency"],
                "retrieval_latency": result["retrieval_latency"],
                "generation_latency": result["generation_latency"],
            },
        }
        output["questions"].append(item)
        if index % args.save_every == 0:
            args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {len(output['questions'])} predictions to {args.output}")


if __name__ == "__main__":
    main()
