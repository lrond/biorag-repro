from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
PUBMED_CACHE_DIR = INTERIM_DIR / "pubmed_cache"
INDEX_DIR = DATA_DIR / "indexes"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"

TRAINING_ZIP = RAW_DIR / "BioASQ-training12b.zip"
GOLDEN_ZIP = RAW_DIR / "Task12BGoldenEnriched.zip"
TRAINING_MEMBER = "BioASQ-training12b/training12b_new.json"

ALL_PROCESSED = PROCESSED_DIR / "all_processed.jsonl"
TRAIN_PROCESSED = PROCESSED_DIR / "train_processed.jsonl"
EVAL_PROCESSED = PROCESSED_DIR / "eval_processed.jsonl"
SPLIT_MANIFEST = PROCESSED_DIR / "split_manifest.json"

BASELINE_PREDICTIONS = OUTPUT_DIR / "predictions_baseline.json"
FULL_PREDICTIONS = OUTPUT_DIR / "predictions.json"
RESULTS_JSON = OUTPUT_DIR / "evaluation_results.json"
RESULTS_MD = OUTPUT_DIR / "evaluation_results.md"
ERROR_REPORT = OUTPUT_DIR / "error_analysis.json"
RUN_MANIFEST = OUTPUT_DIR / "run_manifest.json"
TRAINING_METRICS = OUTPUT_DIR / "training_metrics.json"
TRAINING_CURVE_CSV = OUTPUT_DIR / "training_loss_curve.csv"
TRAINING_CURVE_PNG = OUTPUT_DIR / "training_loss_curve.png"

BASE_RETRIEVER_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
FINETUNED_RETRIEVER_DIR = MODEL_DIR / "bioasq-finetuned-retriever"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"

HOLDOUT_SIZE = 500
RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
TEMPERATURE = 0.05

BASELINE_TOP_K = 3
RERANK_CANDIDATE_K = 20
FINAL_TOP_K = 3
RERANK_BATCH_SIZE = 32
MAX_NEW_TOKENS = 256

PUBMED_BATCH_SIZE = 50
PUBMED_SLEEP_SECONDS = 0.34

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
USE_BM25 = os.getenv("BIORAG_USE_BM25", "0").strip().lower() in {"1", "true", "yes"}


def ensure_dirs() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        INTERIM_DIR,
        PUBMED_CACHE_DIR,
        INDEX_DIR,
        OUTPUT_DIR,
        MODEL_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
