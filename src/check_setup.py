from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from config import (
    BASE_RETRIEVER_MODEL,
    EVAL_PROCESSED,
    GENERATOR_MODEL,
    GOLDEN_ZIP,
    INDEX_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    PUBMED_CACHE_DIR,
    RERANKER_MODEL,
    TRAINING_MEMBER,
    TRAINING_ZIP,
    TRAIN_PROCESSED,
    ensure_dirs,
)


REQUIRED_PACKAGES = {
    "torch": "model training and generation",
    "transformers": "Qwen generator",
    "accelerate": "low-memory model loading",
    "sentence_transformers": "retriever and reranker",
    "faiss": "vector index",
    "numpy": "embedding arrays and metric aggregation",
    "rank_bm25": "BM25 index construction",
    "tqdm": "progress bars",
}

OPTIONAL_PACKAGES = {
    "bert_score": "BERTScore evaluation",
    "matplotlib": "training curve PNG rendering",
}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_zip(path: Path, member: str | None = None) -> tuple[str, str]:
    if not path.exists():
        return "FAIL", f"missing: {path}"
    if path.suffix.lower() != ".zip":
        return "PASS", f"exists: {path}"
    try:
        with ZipFile(path) as archive:
            names = set(archive.namelist())
    except BadZipFile:
        return "FAIL", f"not a readable zip: {path}"
    if member and member not in names:
        return "FAIL", f"{path} does not contain {member}"
    return "PASS", f"exists: {path}"


def add(checks: list[tuple[str, str, str]], status: str, name: str, detail: str) -> None:
    checks.append((status, name, detail))


def collect_checks(device: str) -> list[tuple[str, str, str]]:
    ensure_dirs()
    checks: list[tuple[str, str, str]] = []
    add(checks, "PASS", "Python", f"{platform.python_version()} at {sys.executable}")

    status, detail = check_zip(TRAINING_ZIP, TRAINING_MEMBER)
    add(checks, status, "BioASQ training archive", detail)
    status, detail = check_zip(GOLDEN_ZIP)
    add(checks, status, "BioASQ golden archive", detail)

    for package, reason in REQUIRED_PACKAGES.items():
        add(
            checks,
            "PASS" if module_available(package) else "FAIL",
            f"package:{package}",
            reason,
        )
    for package, reason in OPTIONAL_PACKAGES.items():
        add(
            checks,
            "PASS" if module_available(package) else "WARN",
            f"package:{package}",
            reason,
        )

    if device.startswith("cuda"):
        if module_available("torch"):
            import torch  # type: ignore

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                add(checks, "PASS", "CUDA", f"{name}, {memory_gb:.1f} GB VRAM")
            else:
                add(checks, "FAIL", "CUDA", "device=cuda but torch cannot see a GPU")
        else:
            add(checks, "FAIL", "CUDA", "torch is not installed")
    else:
        add(checks, "PASS", "device", f"configured device is {device}")

    processed_ready = TRAIN_PROCESSED.exists() and EVAL_PROCESSED.exists()
    add(
        checks,
        "PASS" if processed_ready else "WARN",
        "processed data",
        "ready" if processed_ready else "run python src/prepare_data.py first",
    )
    add(checks, "PASS", "PubMed cache", f"{len(list(PUBMED_CACHE_DIR.glob('*.json')))} cached files")
    add(checks, "PASS", "index directory", str(INDEX_DIR))
    add(checks, "PASS", "model directory", str(MODEL_DIR))
    add(checks, "PASS", "output directory", str(OUTPUT_DIR))
    return checks


def print_checks(checks: list[tuple[str, str, str]]) -> None:
    print("BioRAG setup check\n")
    for status, name, detail in checks:
        print(f"[{status}] {name}: {detail}")
    failures = sum(1 for status, _, _ in checks if status == "FAIL")
    warnings = sum(1 for status, _, _ in checks if status == "WARN")
    print(f"\nSummary: {failures} failure(s), {warnings} warning(s).")
    print("\nConfigured models:")
    print(f"- retriever: {BASE_RETRIEVER_MODEL}")
    print(f"- reranker: {RERANKER_MODEL}")
    print(f"- generator: {GENERATOR_MODEL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check BioRAG data, dependencies, and device.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 on failures.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    checks = collect_checks(args.device)
    print_checks(checks)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [{"status": status, "name": name, "detail": detail} for status, name, detail in checks],
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.strict and any(status == "FAIL" for status, _, _ in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
