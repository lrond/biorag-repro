# BioRAG Reproduction Scaffold

[English](README.md) | [简体中文](README.zh-CN.md)

`BioRAG` 的严格复现仓库骨架，默认对齐 `group2.pdf` 中描述的 BioASQ 12b 实验协议，不要求在当前机器直接跑通。

## What Is Included

- 官方 BioASQ Task B zip / JSON 解析与标准化
- `linked_pubmed` 与 `pubmed_dump` 两种语料构建模式
- 报告口径的 `500` 条分层 holdout 评测集，固定 `seed=42`，并保证训练问题与评测问题零重叠
- 冻结 bi-encoder baseline 与对比学习 + cross-encoder 的完整流水线入口
- 更接近官方口径的 BioASQ 指标实现：Yes/No Accuracy / F1，Factoid Strict / Lenient / MRR，List Precision / Recall / F-measure
- 分层配置、运行产物目录、预测文件、评测报告与论文风格汇总表
- 轻量 toy tests 与 CI

## Repository Layout

```text
configs/                 Layered experiment configuration
data/
  raw/                   Optional location for official BioASQ archives
  interim/               Caches and sampled ids
  processed/             Canonical questions/corpus/index metadata
src/biorag/              Runtime package
tests/                   Toy fixtures and smoke tests
```

## Install

Minimal local scaffold usage:

```bash
python3 -m pip install -e .
```

Cloud / GPU usage:

```bash
python3 -m pip install -e '.[cloud]'
```

## Key Commands

If your official archives live under `data/raw/`:

```bash
biorag prepare-data --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name prep --device cpu
biorag build-corpus --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name corpus --device cpu
biorag build-index --config configs/baseline.yaml --input-dir data/raw --output-dir outputs --run-name baseline-index --device cuda
biorag train-retriever --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name train --device cuda
biorag run-baseline --config configs/baseline.yaml --input-dir data/raw --output-dir outputs --run-name baseline --device cuda
biorag run-full-pipeline --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name full --device cuda
biorag-report --baseline-report outputs/baseline/evaluation/evaluation_report.json --ours-report outputs/full/evaluation/evaluation_report.json --output-dir outputs/comparison
```

The default strict dataset config expects these filenames:

- `BioASQ-training12b.zip`
- `Task12BGoldenEnriched.zip`

Every command accepts the same operational flags:

- `--config`
- `--input-dir`
- `--output-dir`
- `--run-name`
- `--device`

The resolved configuration snapshot is saved into the run directory for reproducibility.

## Data Expectations

The project assumes you will place the official BioASQ Task B archives either in the repo root or under `data/raw/`, then point `--input-dir` accordingly. These files are not tracked by git.

By default the strict reproduction uses:

- training source: `BioASQ-training12b.zip`
- auxiliary official test source: `Task12BGoldenEnriched.zip`
- evaluation protocol: `report_holdout`
- corpus mode: `linked_pubmed`
- fixed stratified holdout size: `500`
- holdout seed: `42`
- train/eval question overlap: `0`
- retriever document text: `title + abstract`
- generation decoding: greedy

`report_holdout` matches the report rather than the public challenge leaderboard: the repo draws a reproducible `500`-question stratified evaluation split from `training12b`, trains only on the remaining questions, and still keeps the `golden` archive available for later official-style comparisons.

## Notes On Architecture

- The default configs use the paper-aligned model ids:
  - retriever: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
  - reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - generator: `Qwen/Qwen2.5-7B-Instruct`
- The default dataset preset is [configs/dataset/bioasq12b_report_strict.yaml](configs/dataset/bioasq12b_report_strict.yaml), and both [configs/baseline.yaml](configs/baseline.yaml) and [configs/full_biorag.yaml](configs/full_biorag.yaml) now point to it.
- If you want to evaluate against the official public `golden` batches instead of the report holdout split, switch to [configs/dataset/bioasq12b_golden_eval.yaml](configs/dataset/bioasq12b_golden_eval.yaml).
- Each model block supports two execution modes:
  - `mode: pretrained`: load directly from `model_name`
  - `mode: finetuned`: keep the same base `model_name` but optionally load local weights from `checkpoint_path`
- In `run-full-pipeline`, the retriever is trained first and then reused from the produced checkpoint automatically, so the repo covers both zero-shot and fine-tuned retriever paths out of the box.
- BioASQ document links are normalized from PubMed URLs to raw PMID ids before corpus building, retrieval training, and evaluation.
- Generator loading is session-scoped instead of per-question, and the recorded latency now includes retrieval, reranking, and generation stages.
- CLI execution now seeds Python, NumPy, and Torch from `runtime.seed` before any stage work begins.
- Heavy dependencies are loaded lazily. This keeps toy tests fast while preserving a realistic cloud execution path.
- Config files are valid YAML via JSON subset syntax so they can be parsed without an extra YAML dependency in toy environments.

## Cloud Bring-Up Checklist

1. Install `.[cloud]`.
2. Put `BioASQ-training12b.zip` and `Task12BGoldenEnriched.zip` under `data/raw/`, or keep them in the repo root.
3. Set `NCBI_EMAIL` if you plan to use `linked_pubmed`.
4. Keep `configs/baseline.yaml` and `configs/full_biorag.yaml` as the report-aligned defaults unless you intentionally want a different protocol.
5. Run `prepare-data` first and inspect `outputs/<run>/canonical/split_manifest.json`.
6. Run `build-corpus`, then either `run-baseline` or `run-full-pipeline`.
7. Run `biorag-report` to generate the paper-style comparison table once both reports are available.
