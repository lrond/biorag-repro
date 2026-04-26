from __future__ import annotations

import argparse
import time

from rag_engine import BioRAG
from rag_engine_baseline import BaselineRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure end-to-end latency.")
    parser.add_argument("--mode", choices=["baseline", "full"], default="baseline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--question",
        default="Is Hirschsprung disease a mendelian or a multifactorial disorder?",
    )
    args = parser.parse_args()

    rag = BaselineRAG(device=args.device) if args.mode == "baseline" else BioRAG(device=args.device)
    rag.generate_answer(args.question)
    times = []
    for index in range(args.repeat):
        start = time.perf_counter()
        rag.generate_answer(args.question)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {index + 1}: {elapsed:.2f}s")
    print(f"Average latency: {sum(times) / len(times):.2f}s")


if __name__ == "__main__":
    main()
