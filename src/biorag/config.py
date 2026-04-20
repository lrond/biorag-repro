from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from biorag.utils import deep_merge


class ExperimentConfig(BaseModel):
    name: str
    mode: str


class RuntimeConfig(BaseModel):
    seed: int = 42
    num_workers: int = 0
    log_level: str = "INFO"


class PathsConfig(BaseModel):
    cache_dir: str = "data/interim"
    artifact_dir: str = "outputs"
    processed_dir: str = "data/processed"


class DatasetModeConfig(BaseModel):
    base_url: str = ""
    tool: str = ""
    email_env: str = ""
    batch_size: int = 50
    sleep_seconds: float = 0.34
    input_path: str = ""
    id_field: str = "pmid"
    title_field: str = "title"
    abstract_field: str = "abstract"


class DatasetConfig(BaseModel):
    name: str
    question_path: str
    sample_size: int = 500
    sample_seed: int = 42
    corpus_mode: str = "linked_pubmed"
    cache_dir: str = "data/interim/pubmed_cache"
    canonical_questions_path: str = "data/processed/questions.jsonl"
    sampled_questions_path: str = "data/interim/sampled_question_ids.json"
    linked_pubmed: DatasetModeConfig = Field(default_factory=DatasetModeConfig)
    pubmed_dump: DatasetModeConfig = Field(default_factory=DatasetModeConfig)


class RetrieverConfig(BaseModel):
    backend: str = "transformer"
    model_name: str
    pooling: str = "cls"
    normalize: bool = True
    freeze: bool = False


class RerankerConfig(BaseModel):
    backend: str = "cross_encoder"
    model_name: str


class GeneratorConfig(BaseModel):
    backend: str = "huggingface"
    model_name: str
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0


class ModelsConfig(BaseModel):
    retriever: RetrieverConfig
    reranker: RerankerConfig
    generator: GeneratorConfig


class TrainingConfig(BaseModel):
    enabled: bool = True
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_length: int = 512
    temperature: float = 0.05
    rerank_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    checkpoint_dir: str = "checkpoints/full_biorag"


class InferenceConfig(BaseModel):
    retrieve_top_k: int = 20
    rerank_enabled: bool = True
    rerank_top_k: int = 20
    final_top_k: int = 3
    abstain_when_empty: bool = True
    abstain_threshold: float = 0.05
    max_prompt_characters: int = 6000
    greedy_decoding: bool = True


class EvaluationConfig(BaseModel):
    compute_latency: bool = True
    compute_rouge_l: bool = True
    compute_bert_score: bool = True
    summary_table_name: str = "results_summary.md"
    report_name: str = "evaluation_report.json"


class ProjectConfig(BaseModel):
    experiment: ExperimentConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig
    models: ModelsConfig
    training: TrainingConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    input_dir: str = "data/raw"
    output_dir: str = "outputs"
    run_name: str = "default"
    device: str = "cpu"


def _load_yaml_subset(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return json.loads(text)


def _resolve_config_reference(config_root: Path, reference: str) -> Path:
    candidate = config_root / reference
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".yaml")


def load_project_config(config_path: str | Path) -> ProjectConfig:
    path = Path(config_path).resolve()
    config_root = path.parent
    raw = _load_yaml_subset(path)
    merged: dict[str, Any] = {}
    defaults = raw.get("defaults", {})
    if defaults:
        base_ref = defaults.get("base")
        if base_ref:
            merged = deep_merge(merged, _load_yaml_subset(_resolve_config_reference(config_root, base_ref)))
        for section_name in ("dataset", "models", "training", "inference", "evaluation"):
            reference = defaults.get(section_name)
            if reference:
                merged[section_name] = _load_yaml_subset(
                    _resolve_config_reference(config_root, reference)
                )
    merged = deep_merge(merged, raw.get("overrides", {}))
    if not defaults:
        merged = deep_merge(merged, raw)
    return ProjectConfig.model_validate(merged)


def apply_runtime_overrides(
    config: ProjectConfig,
    *,
    input_dir: str | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    device: str | None = None,
) -> ProjectConfig:
    payload = config.model_dump(mode="python")
    if input_dir is not None:
        payload["input_dir"] = input_dir
        question_path = Path(payload["dataset"]["question_path"])
        if not question_path.is_absolute():
            payload["dataset"]["question_path"] = str(Path(input_dir) / question_path.name)
        dump_path = Path(payload["dataset"]["pubmed_dump"]["input_path"])
        if dump_path and not dump_path.is_absolute():
            payload["dataset"]["pubmed_dump"]["input_path"] = str(Path(input_dir) / dump_path.name)
    if output_dir is not None:
        payload["output_dir"] = output_dir
    if run_name is not None:
        payload["run_name"] = run_name
    if device is not None:
        payload["device"] = device
    return ProjectConfig.model_validate(payload)
