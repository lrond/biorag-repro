from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from config import (
    BASE_RETRIEVER_MODEL,
    BASELINE_PREDICTIONS,
    EVAL_PROCESSED,
    FINETUNED_RETRIEVER_DIR,
    FULL_PREDICTIONS,
    INDEX_DIR,
    RESULTS_MD,
    RUN_MANIFEST,
    TRAINING_CURVE_CSV,
    TRAINING_METRICS,
    TRAIN_PROCESSED,
    ensure_dirs,
)


def run(args: list[str], manifest: list[dict[str, object]], stage: str) -> None:
    command = [sys.executable] + args
    print("+", " ".join(command), flush=True)
    start = time.perf_counter()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        manifest.append(
            {
                "stage": stage,
                "command": command,
                "status": "failed",
                "returncode": exc.returncode,
                "seconds": round(time.perf_counter() - start, 3),
            }
        )
        raise
    else:
        manifest.append(
            {
                "stage": stage,
                "command": command,
                "status": "completed",
                "seconds": round(time.perf_counter() - start, 3),
            }
        )


def stage_done(outputs: list[Path]) -> bool:
    return all(path.exists() and (path.is_dir() or path.stat().st_size > 0) for path in outputs)


def index_ready(model_source: str) -> bool:
    metadata_path = INDEX_DIR / "index_metadata.json"
    if not stage_done([INDEX_DIR / "faiss.index", metadata_path]):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metadata.get("model") == model_source


def run_stage(
    name: str,
    command: list[str],
    outputs: list[Path],
    manifest: list[dict[str, object]],
    *,
    resume: bool,
    force: bool,
    ready: Callable[[], bool] | None = None,
) -> None:
    is_ready = ready() if ready is not None else stage_done(outputs)
    if resume and not force and is_ready:
        print(f"[skip] {name}: outputs already exist", flush=True)
        manifest.append(
            {
                "stage": name,
                "command": [sys.executable] + command,
                "status": "skipped",
                "outputs": [str(path) for path in outputs],
            }
        )
        return
    print(f"[run] {name}", flush=True)
    run(command, manifest, name)


def run_prepare(offline: bool, manifest: list[dict[str, object]], resume: bool, force: bool) -> None:
    prepare = ["src/prepare_data.py"]
    if offline:
        prepare.append("--offline")
    run_stage(
        "prepare_data",
        prepare,
        [TRAIN_PROCESSED, EVAL_PROCESSED],
        manifest,
        resume=resume,
        force=force,
    )


def run_baseline(
    device: str,
    offline: bool,
    manifest: list[dict[str, object]],
    *,
    resume: bool,
    force: bool,
) -> None:
    run_prepare(offline, manifest, resume, force)
    run_stage(
        "build_base_index",
        ["src/build_index.py", "--model", "base", "--device", device],
        [INDEX_DIR / "faiss.index"],
        manifest,
        resume=resume,
        force=force,
        ready=lambda: index_ready(BASE_RETRIEVER_MODEL),
    )
    run_stage(
        "baseline_inference",
        ["src/batch_inference_baseline.py", "--device", device],
        [BASELINE_PREDICTIONS],
        manifest,
        resume=resume,
        force=force,
    )
    run_stage(
        "evaluate",
        ["src/evaluate_official.py"],
        [RESULTS_MD],
        manifest,
        resume=False,
        force=True,
    )


def run_full(
    device: str,
    offline: bool,
    manifest: list[dict[str, object]],
    *,
    resume: bool,
    force: bool,
) -> None:
    run_prepare(offline, manifest, resume, force)
    run_stage(
        "train_retriever",
        ["src/train_retriever.py", "--device", device],
        [FINETUNED_RETRIEVER_DIR, TRAINING_METRICS],
        manifest,
        resume=resume,
        force=force,
    )
    run_stage(
        "build_finetuned_index",
        ["src/build_index.py", "--model", "finetuned", "--device", device],
        [INDEX_DIR / "faiss.index"],
        manifest,
        resume=resume,
        force=force,
        ready=lambda: index_ready(str(FINETUNED_RETRIEVER_DIR)),
    )
    run_stage(
        "full_inference",
        ["src/batch_inference.py", "--device", device],
        [FULL_PREDICTIONS],
        manifest,
        resume=resume,
        force=force,
    )
    run_stage(
        "evaluate",
        ["src/evaluate_official.py"],
        [RESULTS_MD],
        manifest,
        resume=False,
        force=True,
    )
    run_stage(
        "plot_training_curve",
        ["src/plot_training_curve.py"],
        [TRAINING_CURVE_CSV],
        manifest,
        resume=resume,
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BioRAG pipeline.")
    parser.add_argument("--profile", choices=["baseline", "full", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed stages when their outputs already exist.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run stages even if outputs exist.")
    parser.add_argument("--skip-check", action="store_true", help="Skip setup validation.")
    args = parser.parse_args()

    ensure_dirs()
    manifest: list[dict[str, object]] = [
        {
            "profile": args.profile,
            "device": args.device,
            "offline": args.offline,
            "resume": args.resume,
            "force": args.force,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]
    try:
        if not args.skip_check:
            run(["src/check_setup.py", "--device", args.device, "--strict"], manifest, "check_setup")
        if args.profile in {"baseline", "all"}:
            run_baseline(args.device, args.offline, manifest, resume=args.resume, force=args.force)
        if args.profile in {"full", "all"}:
            run_full(args.device, args.offline, manifest, resume=args.resume, force=args.force)
        manifest[0]["status"] = "completed"
    finally:
        manifest[0]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        RUN_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Run manifest saved to {RUN_MANIFEST}")


if __name__ == "__main__":
    main()
