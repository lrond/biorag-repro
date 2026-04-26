from __future__ import annotations

import argparse
import subprocess
import sys


def run(args: list[str]) -> None:
    command = [sys.executable] + args
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_baseline(device: str, offline: bool) -> None:
    prepare = ["src/prepare_data.py"]
    if offline:
        prepare.append("--offline")
    run(prepare)
    run(["src/build_index.py", "--model", "base", "--device", device])
    run(["src/batch_inference_baseline.py", "--device", device])
    run(["src/evaluate_official.py"])


def run_full(device: str, offline: bool) -> None:
    prepare = ["src/prepare_data.py"]
    if offline:
        prepare.append("--offline")
    run(prepare)
    run(["src/train_retriever.py", "--device", device])
    run(["src/build_index.py", "--model", "finetuned", "--device", device])
    run(["src/batch_inference.py", "--device", device])
    run(["src/evaluate_official.py"])
    run(["src/plot_training_curve.py"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cleaned report-version pipeline.")
    parser.add_argument("--profile", choices=["baseline", "full", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.profile in {"baseline", "all"}:
        run_baseline(args.device, args.offline)
    if args.profile in {"full", "all"}:
        run_full(args.device, args.offline)


if __name__ == "__main__":
    main()
