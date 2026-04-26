from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import BASELINE_PREDICTIONS, ERROR_REPORT, EVAL_PROCESSED, FULL_PREDICTIONS
from data_utils import load_jsonl
from evaluate_official import rouge_l


def load_predictions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["id"]: str(item.get("ideal_answer", ""))
        for item in payload.get("questions", [])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze low-scoring BioRAG examples.")
    parser.add_argument("--gt", type=Path, default=EVAL_PROCESSED)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PREDICTIONS)
    parser.add_argument("--full", type=Path, default=FULL_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=ERROR_REPORT)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    gt_rows = load_jsonl(args.gt)
    baseline = load_predictions(args.baseline)
    full = load_predictions(args.full)
    cases = []
    for row in gt_rows:
        reference = " ".join(row.get("ideal_answer", []) or [])
        base_answer = baseline.get(row["id"], "")
        full_answer = full.get(row["id"], "")
        base_score = rouge_l(base_answer, reference) if base_answer else 0.0
        full_score = rouge_l(full_answer, reference) if full_answer else 0.0
        cases.append(
            {
                "id": row["id"],
                "type": row["type"],
                "question": row["body"],
                "reference": reference,
                "baseline_answer": base_answer,
                "biorag_answer": full_answer,
                "baseline_rouge_l": base_score,
                "biorag_rouge_l": full_score,
                "delta": full_score - base_score,
            }
        )
    cases.sort(key=lambda item: item["delta"])
    report = {
        "worst_regressions": cases[: args.limit],
        "best_improvements": list(reversed(cases[-args.limit :])),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved error analysis to {args.output}")


if __name__ == "__main__":
    main()
