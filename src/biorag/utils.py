from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def configure_torch_runtime(runtime_config: Any) -> None:
    allocator_conf = getattr(runtime_config, "cuda_allocator_conf", "")
    if allocator_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", allocator_conf)
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return
    if getattr(runtime_config, "allow_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    matmul_precision = getattr(runtime_config, "matmul_precision", "")
    if matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def chunked_iterable(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


_GREEK_NORMALIZATION = str.maketrans(
    {
        "α": " alpha ",
        "β": " beta ",
        "γ": " gamma ",
        "δ": " delta ",
        "κ": " kappa ",
        "λ": " lambda ",
        "μ": " mu ",
        "ω": " omega ",
    }
)


def normalize_text(text: str) -> str:
    lowered = unicodedata.normalize("NFKD", text).translate(_GREEK_NORMALIZATION).lower().strip()
    lowered = lowered.encode("ascii", "ignore").decode("ascii")
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s\-]", "", lowered)
    return lowered.strip()


def simple_tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]


def extract_pmid(reference: str) -> str:
    text = str(reference).strip()
    if not text:
        return ""
    if text.isdigit():
        return text
    for pattern in (
        r"(?:pubmed|pmid)[/: ]+(\d+)",
        r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)",
        r"/(\d+)(?:[/?#].*)?$",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return text


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def seeded_random(seed: int) -> random.Random:
    return random.Random(seed)
