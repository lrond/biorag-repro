from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from biorag.config import ProjectConfig, QuestionSourceConfig


@dataclass(frozen=True)
class DoctorCheck:
    status: str
    name: str
    detail: str


def _check(status: str, name: str, detail: str) -> DoctorCheck:
    return DoctorCheck(status=status, name=name, detail=detail)


def _has_module(module_name: str) -> bool:
    return find_spec(module_name) is not None


def _resolve_source_path(
    source: QuestionSourceConfig,
    *,
    fallback_path: str,
    input_dir: str,
) -> Path | None:
    raw_path = source.path or fallback_path
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return Path(input_dir) / path.name


def _check_zip_members(
    path: Path,
    expected_members: list[str],
    label: str,
) -> DoctorCheck:
    if not expected_members:
        return _check("PASS", label, f"{path} exists")
    try:
        with ZipFile(path) as archive:
            names = set(archive.namelist())
    except BadZipFile:
        return _check("FAIL", label, f"{path} is not a readable zip archive")
    missing = [member for member in expected_members if member not in names]
    if missing:
        return _check("FAIL", label, f"{path} is missing members: {', '.join(missing)}")
    return _check("PASS", label, f"{path} exists and expected members are present")


def _check_question_source(
    source: QuestionSourceConfig,
    *,
    fallback_path: str,
    input_dir: str,
    label: str,
    required: bool,
) -> DoctorCheck:
    path = _resolve_source_path(source, fallback_path=fallback_path, input_dir=input_dir)
    if path is None:
        status = "FAIL" if required else "WARN"
        return _check(status, label, "no source path configured")
    if not path.exists():
        status = "FAIL" if required else "WARN"
        return _check(status, label, f"{path} does not exist")
    if path.suffix.lower() == ".zip":
        return _check_zip_members(path, source.member_names, label)
    return _check("PASS", label, f"{path} exists")


def _required_modules(config: ProjectConfig) -> list[tuple[str, str]]:
    modules: list[tuple[str, str]] = []
    if config.models.retriever.backend == "transformer":
        modules.extend(
            [
                ("torch", "transformer retriever"),
                ("transformers", "transformer retriever"),
                ("numpy", "transformer retriever"),
            ]
        )
    if config.models.reranker.backend == "cross_encoder":
        modules.append(("sentence_transformers", "cross-encoder reranker"))
    if config.models.generator.backend == "huggingface":
        modules.extend(
            [
                ("torch", "huggingface generator"),
                ("transformers", "huggingface generator"),
                ("accelerate", "low-memory generator loading"),
            ]
        )
    return sorted(set(modules))


def _optional_modules(config: ProjectConfig) -> list[tuple[str, str]]:
    modules = [("faiss", "fast vector search"), ("matplotlib", "training curve plotting")]
    if config.evaluation.compute_bert_score:
        modules.append(("bert_score", "BERTScore evaluation"))
    return sorted(set(modules))


def _merge_module_reasons(modules: list[tuple[str, str]]) -> list[tuple[str, str]]:
    reasons_by_module: dict[str, list[str]] = {}
    for module_name, reason in modules:
        reasons_by_module.setdefault(module_name, [])
        if reason not in reasons_by_module[module_name]:
            reasons_by_module[module_name].append(reason)
    return [
        (module_name, ", ".join(reasons))
        for module_name, reasons in sorted(reasons_by_module.items())
    ]


def collect_diagnostics(config: ProjectConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    python_version = sys.version_info
    if python_version >= (3, 11):
        checks.append(
            _check(
                "PASS",
                "Python",
                f"{platform.python_version()} at {Path(sys.executable)}",
            )
        )
    else:
        checks.append(
            _check(
                "FAIL",
                "Python",
                f"{platform.python_version()} found; pyproject requires Python >= 3.11",
            )
        )

    input_dir = Path(config.input_dir)
    checks.append(
        _check(
            "PASS" if input_dir.exists() else "FAIL",
            "Input directory",
            f"{input_dir} {'exists' if input_dir.exists() else 'does not exist'}",
        )
    )

    output_dir = Path(config.output_dir)
    output_parent = output_dir if output_dir.exists() else output_dir.parent
    checks.append(
        _check(
            "PASS" if output_parent.exists() else "WARN",
            "Output directory",
            f"{output_dir} will be used; parent {output_parent} "
            f"{'exists' if output_parent.exists() else 'does not exist yet'}",
        )
    )

    checks.append(
        _check_question_source(
            config.dataset.training,
            fallback_path=config.dataset.question_path,
            input_dir=config.input_dir,
            label="BioASQ training source",
            required=True,
        )
    )

    golden_required = config.dataset.protocol == "golden_eval"
    checks.append(
        _check_question_source(
            config.dataset.golden,
            fallback_path="",
            input_dir=config.input_dir,
            label="BioASQ golden source",
            required=golden_required,
        )
    )

    if config.dataset.corpus_mode == "pubmed_dump":
        dump_path = Path(config.dataset.pubmed_dump.input_path)
        if not dump_path.is_absolute() and not dump_path.exists():
            dump_path = Path(config.input_dir) / dump_path.name
        checks.append(
            _check(
                "PASS" if dump_path.exists() else "FAIL",
                "PubMed dump",
                f"{dump_path} {'exists' if dump_path.exists() else 'does not exist'}",
            )
        )

    if config.dataset.corpus_mode == "linked_pubmed":
        email_env = config.dataset.linked_pubmed.email_env
        detail = (
            f"{email_env} is set"
            if os.getenv(email_env)
            else f"{email_env} is not set; NCBI requests may be slower or rejected"
        )
        checks.append(_check("PASS" if os.getenv(email_env) else "WARN", "NCBI email", detail))

    for module_name, reason in _merge_module_reasons(_required_modules(config)):
        checks.append(
            _check(
                "PASS" if _has_module(module_name) else "FAIL",
                f"Package {module_name}",
                f"required for {reason}",
            )
        )

    for module_name, reason in _merge_module_reasons(_optional_modules(config)):
        checks.append(
            _check(
                "PASS" if _has_module(module_name) else "WARN",
                f"Package {module_name}",
                f"used for {reason}",
            )
        )

    device = config.device
    if device.startswith("cuda"):
        if not _has_module("torch"):
            checks.append(_check("FAIL", "CUDA", "torch is not installed"))
        else:
            import torch  # type: ignore

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                checks.append(_check("PASS", "CUDA", f"{name}, {memory_gb:.1f} GB VRAM"))
            else:
                checks.append(_check("FAIL", "CUDA", "device=cuda but torch cannot see a GPU"))
    else:
        checks.append(_check("PASS", "Device", f"configured device is {device}"))

    return checks


def render_diagnostics(checks: list[DoctorCheck]) -> str:
    lines = ["BioRAG doctor", ""]
    for check in checks:
        lines.append(f"[{check.status}] {check.name}: {check.detail}")
    failures = sum(1 for check in checks if check.status == "FAIL")
    warnings = sum(1 for check in checks if check.status == "WARN")
    lines.extend(
        [
            "",
            f"Summary: {failures} failure(s), {warnings} warning(s).",
        ]
    )
    return "\n".join(lines)


def run_doctor(config: ProjectConfig) -> bool:
    checks = collect_diagnostics(config)
    print(render_diagnostics(checks))
    return not any(check.status == "FAIL" for check in checks)
