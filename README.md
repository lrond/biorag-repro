# BioRAG Reproduction Scaffold

`BioRAG` 的本地复现仓库骨架，面向云端训练与评测，不要求在当前机器直接跑通。

## What Is Included

- 官方 BioASQ Task B JSON 解析与标准化
- `linked_pubmed` 与 `pubmed_dump` 两种语料构建模式
- 冻结 bi-encoder baseline 与对比学习 + cross-encoder 的完整流水线入口
- 分层配置、运行产物目录、预测文件、评测报告与论文风格汇总表
- 轻量 toy tests 与 CI

## Repository Layout

```text
configs/                 Layered experiment configuration
data/
  raw/                   Official BioASQ files live here locally
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

```bash
biorag prepare-data --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name prep --device cpu
biorag build-corpus --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name corpus --device cpu
biorag build-index --config configs/baseline.yaml --input-dir data/raw --output-dir outputs --run-name baseline-index --device cuda
biorag train-retriever --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name train --device cuda
biorag run-baseline --config configs/baseline.yaml --input-dir data/raw --output-dir outputs --run-name baseline --device cuda
biorag run-full-pipeline --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name full --device cuda
```

Every command accepts the same operational flags:

- `--config`
- `--input-dir`
- `--output-dir`
- `--run-name`
- `--device`

The resolved configuration snapshot is saved into the run directory for reproducibility.

## Data Expectations

The project assumes you will later place the official BioASQ Task B files into `data/raw/` or point the dataset config at another location. These files are not tracked by git.

By default the reproduction uses:

- question source: BioASQ 12b Task B JSON
- corpus mode: `linked_pubmed`
- fixed stratified sample size: `500`
- sample seed: `42`
- retriever document text: `title + abstract`
- generation decoding: greedy

## Notes On Architecture

- The default configs use the paper-aligned model ids:
  - retriever: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
  - reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - generator: `Qwen/Qwen2.5-7B-Instruct`
- Heavy dependencies are loaded lazily. This keeps toy tests fast while preserving a realistic cloud execution path.
- Config files are valid YAML via JSON subset syntax so they can be parsed without an extra YAML dependency in toy environments.

## Cloud Bring-Up Checklist

1. Install `.[cloud]`.
2. Put official BioASQ files under `data/raw/`.
3. Set `NCBI_EMAIL` if you plan to use `linked_pubmed`.
4. Adjust `configs/*` if your cloud runner uses different paths or model ids.
5. Run `prepare-data`, `build-corpus`, and then either `run-baseline` or `run-full-pipeline`.
