from __future__ import annotations

import argparse
import json

from config import (
    ALL_PROCESSED,
    EVAL_PROCESSED,
    HOLDOUT_SIZE,
    RANDOM_SEED,
    SPLIT_MANIFEST,
    TRAIN_PROCESSED,
    ensure_dirs,
)
from data_utils import (
    all_pmids,
    build_processed_record,
    load_or_fetch_pubmed,
    read_bioasq_training,
    stratified_holdout,
    type_counts,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BioASQ data.")
    parser.add_argument("--holdout-size", type=int, default=HOLDOUT_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use existing PubMed cache and snippet fallbacks only.",
    )
    args = parser.parse_args()

    ensure_dirs()
    questions = read_bioasq_training()
    train_questions, eval_questions = stratified_holdout(
        questions,
        sample_size=args.holdout_size,
        seed=args.seed,
    )
    pmids = all_pmids(questions)
    documents = load_or_fetch_pubmed(pmids, offline=args.offline)

    processed_all = [build_processed_record(question, documents) for question in questions]
    eval_ids = {str(question.get("id", "")) for question in eval_questions}
    train_ids = {str(question.get("id", "")) for question in train_questions}
    processed_train = [row for row in processed_all if row["id"] in train_ids]
    processed_eval = [row for row in processed_all if row["id"] in eval_ids]

    write_jsonl(ALL_PROCESSED, processed_all)
    write_jsonl(TRAIN_PROCESSED, processed_train)
    write_jsonl(EVAL_PROCESSED, processed_eval)

    manifest = {
        "seed": args.seed,
        "holdout_size": args.holdout_size,
        "all_questions": len(processed_all),
        "train_questions": len(processed_train),
        "eval_questions": len(processed_eval),
        "all_type_counts": type_counts(questions),
        "train_type_counts": type_counts(processed_train),
        "eval_type_counts": type_counts(processed_eval),
        "requested_pmids": len(pmids),
        "resolved_pubmed_documents": len(documents),
        "eval_ids": sorted(eval_ids),
        "question_overlap": len(train_ids & eval_ids),
    }
    SPLIT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
