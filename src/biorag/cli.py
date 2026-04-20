from __future__ import annotations

import argparse
from typing import Callable

from biorag.config import apply_runtime_overrides, load_project_config
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
from biorag.utils import configure_logging


CommandFn = Callable[[object], dict]


def _add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--config", required=True, help="Path to layered experiment config.")
    subparser.add_argument("--input-dir", default=None, help="Root input directory override.")
    subparser.add_argument("--output-dir", default=None, help="Root output directory override.")
    subparser.add_argument("--run-name", default=None, help="Logical run name for output artifacts.")
    subparser.add_argument("--device", default=None, help="Execution device, e.g. cpu, cuda, mps.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BioRAG reproduction command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
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


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_project_config(args.config)
    config = apply_runtime_overrides(
        config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        run_name=args.run_name or config.experiment.name,
        device=args.device or "cpu",
    )
    configure_logging(config.runtime.log_level)
    save_config_snapshot(config)
    handler = _dispatch(args.command)
    handler(config)
