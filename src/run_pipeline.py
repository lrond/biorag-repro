from __future__ import annotations

import argparse
import json
import os
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
    LOG_DIR,
    RESULTS_MD,
    RUN_MANIFEST,
    TRAINING_CURVE_CSV,
    TRAINING_METRICS,
    TRAIN_PROCESSED,
    ensure_dirs,
)


def format_duration(seconds: float) -> str:
    seconds_int = int(seconds)
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def command_text(command: list[str]) -> str:
    return " ".join(command)


class PipelineReporter:
    def __init__(self, total_stages: int, log_path: Path, stream_output: bool) -> None:
        self.total_stages = total_stages
        self.log_path = log_path
        self.stream_output = stream_output
        self.current_stage = 0
        self.started_at = time.perf_counter()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8", buffering=1)

    def write(self, message: str = "") -> None:
        print(message, flush=True)
        self.log_file.write(message + "\n")

    def write_child_output(self, text: str) -> None:
        self.log_file.write(text)
        if self.stream_output:
            print(text, end="", flush=True)

    def banner(self, profile: str, device: str, resume: bool, force: bool, offline: bool) -> None:
        self.write("=" * 72)
        self.write("BioRAG pipeline")
        self.write(f"profile: {profile} | device: {device} | resume: {resume} | force: {force} | offline: {offline}")
        self.write(f"log file: {self.log_path}")
        self.write("=" * 72)

    def start_stage(self, stage: str, command: list[str]) -> float:
        self.current_stage += 1
        self.write("")
        self.write("=" * 72)
        self.write(f"[{self.current_stage}/{self.total_stages}] {stage}")
        self.write(f"command: {command_text(command)}")
        self.write("-" * 72)
        return time.perf_counter()

    def finish_stage(self, stage: str, status: str, seconds: float) -> None:
        self.write("")
        self.write(f"[{status}] {stage} ({format_duration(seconds)})")

    def skip_stage(self, stage: str, command: list[str], outputs: list[Path]) -> None:
        self.current_stage += 1
        self.write("")
        self.write("=" * 72)
        self.write(f"[{self.current_stage}/{self.total_stages}] {stage}")
        self.write(f"command: {command_text(command)}")
        self.write("[skip] outputs already exist")
        for output in outputs:
            self.write(f"  - {output}")

    def summary(self, manifest: list[dict[str, object]], manifest_path: Path) -> None:
        total_seconds = time.perf_counter() - self.started_at
        self.write("")
        self.write("=" * 72)
        self.write("Pipeline summary")
        for record in manifest[1:]:
            stage = record.get("stage", "unknown")
            status = record.get("status", "unknown")
            seconds = record.get("seconds")
            if isinstance(seconds, (int, float)):
                self.write(f"- {stage}: {status} in {format_duration(float(seconds))}")
            else:
                self.write(f"- {stage}: {status}")
        self.write(f"total time: {format_duration(total_seconds)}")
        self.write(f"manifest: {manifest_path}")
        self.write(f"log file: {self.log_path}")
        self.write("=" * 72)

    def close(self) -> None:
        self.log_file.close()


def run(args: list[str], manifest: list[dict[str, object]], stage: str, reporter: PipelineReporter) -> None:
    command = [sys.executable] + args
    start = reporter.start_stage(stage, command)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            reporter.write_child_output(line)
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        seconds = time.perf_counter() - start
        manifest.append(
            {
                "stage": stage,
                "command": command,
                "status": "interrupted",
                "seconds": round(seconds, 3),
            }
        )
        reporter.finish_stage(stage, "interrupted", seconds)
        raise

    seconds = time.perf_counter() - start
    if returncode:
        manifest.append(
            {
                "stage": stage,
                "command": command,
                "status": "failed",
                "returncode": returncode,
                "seconds": round(seconds, 3),
            }
        )
        reporter.finish_stage(stage, "failed", seconds)
        raise subprocess.CalledProcessError(returncode, command)

    manifest.append(
        {
            "stage": stage,
            "command": command,
            "status": "completed",
            "seconds": round(seconds, 3),
        }
    )
    reporter.finish_stage(stage, "done", seconds)


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
    reporter: PipelineReporter,
    *,
    resume: bool,
    force: bool,
    ready: Callable[[], bool] | None = None,
) -> None:
    is_ready = ready() if ready is not None else stage_done(outputs)
    if resume and not force and is_ready:
        full_command = [sys.executable] + command
        reporter.skip_stage(name, full_command, outputs)
        manifest.append(
            {
                "stage": name,
                "command": full_command,
                "status": "skipped",
                "outputs": [str(path) for path in outputs],
            }
        )
        return
    run(command, manifest, name, reporter)


def run_prepare(
    offline: bool,
    manifest: list[dict[str, object]],
    reporter: PipelineReporter,
    resume: bool,
    force: bool,
) -> None:
    prepare = ["src/prepare_data.py"]
    if offline:
        prepare.append("--offline")
    run_stage(
        "prepare_data",
        prepare,
        [TRAIN_PROCESSED, EVAL_PROCESSED],
        manifest,
        reporter,
        resume=resume,
        force=force,
    )


def run_baseline(
    device: str,
    offline: bool,
    manifest: list[dict[str, object]],
    reporter: PipelineReporter,
    *,
    resume: bool,
    force: bool,
) -> None:
    run_prepare(offline, manifest, reporter, resume, force)
    run_stage(
        "build_base_index",
        ["src/build_index.py", "--model", "base", "--device", device],
        [INDEX_DIR / "faiss.index"],
        manifest,
        reporter,
        resume=resume,
        force=force,
        ready=lambda: index_ready(BASE_RETRIEVER_MODEL),
    )
    run_stage(
        "baseline_inference",
        ["src/batch_inference_baseline.py", "--device", device],
        [BASELINE_PREDICTIONS],
        manifest,
        reporter,
        resume=resume,
        force=force,
    )
    run_stage(
        "evaluate",
        ["src/evaluate_official.py"],
        [RESULTS_MD],
        manifest,
        reporter,
        resume=False,
        force=True,
    )


def run_full(
    device: str,
    offline: bool,
    manifest: list[dict[str, object]],
    reporter: PipelineReporter,
    *,
    resume: bool,
    force: bool,
) -> None:
    run_prepare(offline, manifest, reporter, resume, force)
    run_stage(
        "train_retriever",
        ["src/train_retriever.py", "--device", device],
        [FINETUNED_RETRIEVER_DIR, TRAINING_METRICS],
        manifest,
        reporter,
        resume=resume,
        force=force,
    )
    run_stage(
        "build_finetuned_index",
        ["src/build_index.py", "--model", "finetuned", "--device", device],
        [INDEX_DIR / "faiss.index"],
        manifest,
        reporter,
        resume=resume,
        force=force,
        ready=lambda: index_ready(str(FINETUNED_RETRIEVER_DIR)),
    )
    run_stage(
        "full_inference",
        ["src/batch_inference.py", "--device", device],
        [FULL_PREDICTIONS],
        manifest,
        reporter,
        resume=resume,
        force=force,
    )
    run_stage(
        "evaluate",
        ["src/evaluate_official.py"],
        [RESULTS_MD],
        manifest,
        reporter,
        resume=False,
        force=True,
    )
    run_stage(
        "plot_training_curve",
        ["src/plot_training_curve.py"],
        [TRAINING_CURVE_CSV],
        manifest,
        reporter,
        resume=resume,
        force=force,
    )


def planned_stage_count(profile: str, skip_check: bool) -> int:
    count = 0 if skip_check else 1
    if profile in {"baseline", "all"}:
        count += 4
    if profile in {"full", "all"}:
        count += 6
    return count


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
    parser.add_argument("--log-file", type=Path, default=None, help="Terminal output is also saved here.")
    parser.add_argument(
        "--no-stream-output",
        action="store_true",
        help="Show only stage headers and summary, while still saving full subprocess output to the log file.",
    )
    args = parser.parse_args()

    ensure_dirs()
    log_path = args.log_file or LOG_DIR / f"run_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    reporter = PipelineReporter(
        planned_stage_count(args.profile, args.skip_check),
        log_path,
        stream_output=not args.no_stream_output,
    )
    manifest: list[dict[str, object]] = [
        {
            "profile": args.profile,
            "device": args.device,
            "offline": args.offline,
            "resume": args.resume,
            "force": args.force,
            "log_file": str(log_path),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]
    try:
        reporter.banner(args.profile, args.device, args.resume, args.force, args.offline)
        if not args.skip_check:
            run(["src/check_setup.py", "--device", args.device, "--strict"], manifest, "check_setup", reporter)
        if args.profile in {"baseline", "all"}:
            run_baseline(args.device, args.offline, manifest, reporter, resume=args.resume, force=args.force)
        if args.profile in {"full", "all"}:
            run_full(args.device, args.offline, manifest, reporter, resume=args.resume, force=args.force)
        manifest[0]["status"] = "completed"
    except KeyboardInterrupt:
        manifest[0]["status"] = "interrupted"
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        manifest[0]["status"] = "failed"
        manifest[0]["returncode"] = exc.returncode
        raise SystemExit(exc.returncode)
    finally:
        manifest[0]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        RUN_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        reporter.summary(manifest, RUN_MANIFEST)
        reporter.close()


if __name__ == "__main__":
    main()
