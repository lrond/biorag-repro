# BioRAG

[English](README.md) | [简体中文](README.zh-CN.md)

BioRAG 是一个面向 BioASQ 风格生物医学问答任务的
RAG（检索增强生成）项目。仓库内已经包含数据准备、语料构建、索引生成、
检索器训练、重排、答案生成、评测与结果对照表生成等完整流程。

这个项目的目标是让你既可以在云端一键启动完整流水线，也可以在需要的时候
按阶段分别控制每一步。

## 项目特性

- 支持官方 BioASQ 压缩包直接读取与统一问题格式标准化
- 支持两种语料模式：`linked_pubmed` 和 `pubmed_dump`
- 内置两种端到端流程：
  - `baseline`：冻结 bi-encoder，直接检索后生成
  - `full`：先训练 retriever，再重排并生成
- 每个模型模块都支持 `pretrained` 和 `finetuned` 两种加载方式
- 内置 BioASQ 风格评测指标与对照表生成工具
- 目录结构清晰，适合云端实验和中间产物追踪

## 快速开始

1. 安装项目：

```bash
python3 -m pip install -e '.[cloud]'
```

2. 把官方 BioASQ 压缩包放到 `data/raw/`：

- `BioASQ-training12b.zip`
- `Task12BGoldenEnriched.zip`

3. 如果你使用 `linked_pubmed`，先设置 NCBI 邮箱：

```bash
export NCBI_EMAIL="you@example.com"
```

4. 直接运行内置快速入口：

```bash
biorag quickstart --profile baseline --device cuda
biorag quickstart --profile full --device cuda
```

`quickstart` 默认会从 `data/raw/` 读取数据，把结果写到 `outputs/`，
并自动把运行名设为 `baseline` 或 `full`。

如果你的路径不同，可以直接覆盖：

```bash
biorag quickstart \
  --profile full \
  --input-dir /path/to/data \
  --output-dir /path/to/outputs \
  --run-name exp-full \
  --device cuda
```

## 主要入口

大多数情况下，你只需要记住这几个命令：

- `biorag quickstart --profile baseline`
  运行完整的 baseline 流水线。
- `biorag quickstart --profile full`
  运行 retriever 训练、检索、重排、生成和评测的完整流程。
- `biorag-report --baseline-report ... --ours-report ... --output-dir ...`
  生成 `Baseline / Ours / Gain` 风格的对照表。

如果你希望手动控制各个阶段，也可以使用这些命令：

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

这些阶段命令都支持统一参数：

- `--config`
- `--input-dir`
- `--output-dir`
- `--run-name`
- `--device`

## 预设配置

主流程配置：

- [configs/baseline.yaml](configs/baseline.yaml)
- [configs/full_biorag.yaml](configs/full_biorag.yaml)

数据协议配置：

- [configs/dataset/bioasq12b_report_strict.yaml](configs/dataset/bioasq12b_report_strict.yaml)
- [configs/dataset/bioasq12b_golden_eval.yaml](configs/dataset/bioasq12b_golden_eval.yaml)

每个模型模块都支持两种加载模式：

- `mode: pretrained`
  直接从 `model_name` 加载模型
- `mode: finetuned`
  先加载相同的基础 `model_name`，再从 `checkpoint_path` 加载本地权重

默认模型：

- Retriever:
  `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Reranker:
  `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generator:
  `Qwen/Qwen2.5-7B-Instruct`

## 默认数据行为

默认数据预设会查找以下文件：

- `BioASQ-training12b.zip`
- `Task12BGoldenEnriched.zip`

内置协议默认采用：

- 固定 `500` 条分层 holdout
- `seed=42`
- 训练问题与评测问题零重叠
- 默认语料模式为 `linked_pubmed`
- 文档文本使用 `title + abstract`
- 生成阶段使用 greedy decoding

## 输出目录

每个运行目录 `outputs/<run-name>/` 下通常会包含：

- `resolved_config.json`
- `canonical/`
- `corpus/`
- `index/`
- `training/`
- `retrieval/`
- `rerank/`
- `predictions/`
- `evaluation/`

这样既方便排查问题，也方便在云端继续接着跑。

## 仓库结构

```text
configs/                 分层实验配置
data/
  raw/                   官方 BioASQ 压缩包放置目录
  interim/               缓存与中间文件
  processed/             规范化问题、语料与索引元数据
scripts/                 云端与辅助脚本
src/biorag/              主代码
tests/                   toy fixtures 与 smoke tests
```

## 示例流程

如果你想按阶段执行，而不是直接用 `quickstart`，一个常见流程如下：

```bash
biorag prepare-data --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name prep --device cpu
biorag build-corpus --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name corpus --device cpu
biorag run-baseline --config configs/baseline.yaml --input-dir data/raw --output-dir outputs --run-name baseline --device cuda
biorag run-full-pipeline --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name full --device cuda
biorag-report --baseline-report outputs/baseline/evaluation/evaluation_report.json --ours-report outputs/full/evaluation/evaluation_report.json --output-dir outputs/comparison
```

## 开发与检查

安装开发依赖：

```bash
python3 -m pip install -e '.[dev]'
```

运行检查：

```bash
ruff check .
python3 -m unittest discover -s tests -v
```
