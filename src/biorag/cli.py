from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from biorag.config import apply_runtime_overrides, load_project_config
from biorag.doctor import run_doctor
from biorag.pipeline import (
    build_corpus_stage,
    build_index_stage,
    evaluate_stage,
    generate_stage,
    prepare_data,
    rerank_stage,
    retrieve_stage,
    run_baseline,
    run_full_pipeline,
    save_config_snapshot,
    train_retriever_stage,
)
from biorag.utils import configure_logging, configure_torch_runtime, set_global_seed

CommandFn = Callable[[object], dict]
QUICKSTART_PROFILES = {
    "baseline": "configs/baseline.yaml",
    "full": "configs/full_biorag.yaml",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_runtime_arguments(
    subparser: argparse.ArgumentParser,
    *,
    input_dir_default: str | None = None,
    output_dir_default: str | None = None,
) -> None:
    subparser.add_argument(
        "--input-dir",
        default=input_dir_default,
        help="Root input directory override.",
    )
    subparser.add_argument(
        "--output-dir",
        default=output_dir_default,
        help="Root output directory override.",
    )
    subparser.add_argument(
        "--run-name",
        default=None,
        help="Logical run name for output artifacts.",
    )
    subparser.add_argument(
        "--device",
        default=None,
        help="Execution device, e.g. cpu, cuda, mps.",
    )


def _add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--config",
        required=True,
        help="Path to layered experiment config.",
    )
    _add_runtime_arguments(subparser)


def _add_quickstart_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--profile",
        choices=sorted(QUICKSTART_PROFILES),
        default="full",
        help="Quickstart preset to run. 'baseline' skips reranking and retriever training.",
    )
    subparser.add_argument(
        "--config",
        default=None,
        help="Optional custom config override. Defaults to the config preset for the profile.",
    )
    _add_runtime_arguments(
        subparser,
        input_dir_default="data/raw",
        output_dir_default="outputs",
    )


def _add_doctor_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--profile",
        choices=sorted(QUICKSTART_PROFILES),
        default="full",
        help="Preset to validate when --config is not supplied.",
    )
    subparser.add_argument(
        "--config",
        default=None,
        help="Optional custom config to validate.",
    )
    subparser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when any required check fails.",
    )
    _add_runtime_arguments(
        subparser,
        input_dir_default="data/raw",
        output_dir_default="outputs",
    )


def _default_quickstart_config(profile: str) -> Path:
    return _project_root() / QUICKSTART_PROFILES[profile]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BioRAG command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_quickstart_arguments(
        subparsers.add_parser(
            "quickstart",
            help="Run the most common end-to-end pipeline with project defaults.",
        )
    )
    _add_doctor_arguments(
        subparsers.add_parser(
            "doctor",
            help="Validate local data, dependencies, and device readiness.",
        )
    )
    for command in (
        "prepare-data",
        "build-corpus",
        "build-index",
        "train-retriever",
        "retrieve",
        "rerank",
        "generate",
        "evaluate",
        "run-baseline",
        "run-full-pipeline",
    ):
        _add_shared_arguments(subparsers.add_parser(command))
    return parser


def _dispatch(command: str) -> CommandFn:
    mapping: dict[str, CommandFn] = {
        "prepare-data": prepare_data,
        "build-corpus": build_corpus_stage,
        "build-index": build_index_stage,
        "train-retriever": train_retriever_stage,
        "retrieve": retrieve_stage,
        "rerank": rerank_stage,
        "generate": generate_stage,
        "evaluate": evaluate_stage,
        "run-baseline": run_baseline,
        "run-full-pipeline": run_full_pipeline,
    }
    return mapping[command]


def _resolve_command_and_config(
    args: argparse.Namespace,
) -> tuple[str, str, str | None]:
    if args.command != "quickstart":
        if args.command == "doctor":
            config_path = args.config or str(_default_quickstart_config(args.profile))
            return "doctor", config_path, args.profile
        return args.command, args.config, None
    config_path = args.config or str(_default_quickstart_config(args.profile))
    if args.profile == "baseline":
        return "run-baseline", config_path, "baseline"
    return "run-full-pipeline", config_path, "full"


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command, config_path, default_run_name = _resolve_command_and_config(args)
    config = load_project_config(config_path)
    config = apply_runtime_overrides(
        config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        run_name=args.run_name or default_run_name or config.experiment.name,
        device=args.device or "cpu",
    )
    configure_logging(config.runtime.log_level)
    configure_torch_runtime(config.runtime)
    set_global_seed(config.runtime.seed)
    save_config_snapshot(config)
    if command == "doctor":
        ok = run_doctor(config)
        if args.strict and not ok:
            raise SystemExit(1)
        return
    handler = _dispatch(command)
    handler(config)
