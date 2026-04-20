# BioRAG

[English](README.md) | [简体中文](README.zh-CN.md)

BioRAG is a biomedical retrieval-augmented generation pipeline for BioASQ-style
question answering. The repository covers data preparation, corpus building,
indexing, retriever training, reranking, answer generation, evaluation, and
comparison reporting in one place.

It is designed to be easy to launch on a cloud machine while still keeping the
individual stages accessible when you want tighter control.

## Highlights

- Official BioASQ archive ingestion and canonical question normalization
- Two corpus modes: `linked_pubmed` and `pubmed_dump`
- Two end-to-end presets:
  - `baseline`: frozen bi-encoder retrieval and direct generation
  - `full`: contrastive retriever training, reranking, and generation
- Module-level model switching with `pretrained` and `finetuned` modes
- Built-in BioASQ-style metrics and a comparison table generator
- Cloud-friendly run structure with config snapshots and stage outputs

## Quick Start

1. Install the project:

```bash
python3 -m pip install -e '.[cloud]'
```

2. Put the official BioASQ archives under `data/raw/`:

- `BioASQ-training12b.zip`
- `Task12BGoldenEnriched.zip`

3. If you plan to use `linked_pubmed`, set your NCBI email:

```bash
export NCBI_EMAIL="you@example.com"
```

4. Run one of the built-in entry points:

```bash
biorag quickstart --profile baseline --device cuda
biorag quickstart --profile full --device cuda
```

By default, `quickstart` reads from `data/raw/`, writes to `outputs/`, and uses
`baseline` or `full` as the run name.

If your paths are different, override them directly:

```bash
biorag quickstart \
  --profile full \
  --input-dir /path/to/data \
  --output-dir /path/to/outputs \
  --run-name exp-full \
  --device cuda
```

## Main Entry Points

For most users, these are the only commands you need:

- `biorag quickstart --profile baseline`
  Runs the end-to-end baseline pipeline.
- `biorag quickstart --profile full`
  Runs retriever training plus the full retrieval, reranking, generation, and
  evaluation flow.
- `biorag-report --baseline-report ... --ours-report ... --output-dir ...`
  Builds a paper-style `Baseline / Ours / Gain` comparison table.

The stage-by-stage commands remain available when you want more control:

- `prepare-data`
- `build-corpus`
- `build-index`
- `train-retriever`
- `retrieve`
- `rerank`
- `generate`
- `evaluate`
- `run-baseline`
- `run-full-pipeline`

All stage commands accept the same runtime flags:

- `--config`
- `--input-dir`
- `--output-dir`
- `--run-name`
- `--device`

## Preset Configurations

Main project presets:

- [configs/baseline.yaml](configs/baseline.yaml)
- [configs/full_biorag.yaml](configs/full_biorag.yaml)

Dataset presets:

- [configs/dataset/bioasq12b_report_strict.yaml](configs/dataset/bioasq12b_report_strict.yaml)
- [configs/dataset/bioasq12b_golden_eval.yaml](configs/dataset/bioasq12b_golden_eval.yaml)

Each model block supports two loading modes:

- `mode: pretrained`
  Load directly from `model_name`
- `mode: finetuned`
  Load the same base `model_name` and then apply local weights from
  `checkpoint_path`

Default model ids:

- Retriever:
  `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Reranker:
  `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generator:
  `Qwen/Qwen2.5-7B-Instruct`

## Default Data Behavior

The default dataset preset expects:

- `BioASQ-training12b.zip`
- `Task12BGoldenEnriched.zip`

The built-in protocol uses:

- a fixed `500`-question stratified holdout
- `seed=42`
- zero overlap between training and evaluation questions
- `linked_pubmed` as the default corpus mode
- `title + abstract` as the document text
- greedy decoding for generation

## Output Structure

Each run directory under `outputs/<run-name>/` can include:

- `resolved_config.json`
- `canonical/`
- `corpus/`
- `index/`
- `training/`
- `retrieval/`
- `rerank/`
- `predictions/`
- `evaluation/`

This makes it easy to inspect intermediate artifacts or resume work in the
cloud.

## Repository Layout

```text
configs/                 Layered experiment configuration
data/
  raw/                   Optional location for official BioASQ archives
  interim/               Cached ids and intermediate files
  processed/             Canonical questions, corpus, and index metadata
scripts/                 Helper scripts for cloud setup
src/biorag/              Runtime package
tests/                   Toy fixtures and smoke tests
```

## Example Workflow

If you want explicit stage control instead of `quickstart`, a common flow is:

```bash
biorag prepare-data --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name prep --device cpu
biorag build-corpus --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name corpus --device cpu
biorag run-baseline --config configs/baseline.yaml --input-dir data/raw --output-dir outputs --run-name baseline --device cuda
biorag run-full-pipeline --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name full --device cuda
biorag-report --baseline-report outputs/baseline/evaluation/evaluation_report.json --ours-report outputs/full/evaluation/evaluation_report.json --output-dir outputs/comparison
```

## Development

Install dev tools:

```bash
python3 -m pip install -e '.[dev]'
```

Run checks:

```bash
ruff check .
python3 -m unittest discover -s tests -v
```
