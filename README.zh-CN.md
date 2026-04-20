# BioRAG 严格复现仓库

[English](README.md) | [简体中文](README.zh-CN.md)

这个仓库用于严格复现 `group2.pdf` 中描述的 `BioRAG` 实验框架，默认对齐 BioASQ 12b 的报告口径。当前目标是把代码架构、数据协议、实验配置和结果产物准备完整，方便直接迁移到云端运行；不要求在本地机器上真正训练或推理完成。

## 包含内容

- 官方 BioASQ Task B `zip / JSON` 数据解析与统一 schema 标准化
- `linked_pubmed` 和 `pubmed_dump` 两种语料构建模式
- 报告口径的 `500` 条分层 holdout 评测集，固定 `seed=42`
- 训练集与评测集问题零重叠的数据切分流程
- `baseline` 与 `full_biorag` 两套完整流水线
- 更接近官方 BioASQ 口径的评测实现
- 论文风格的 `Baseline vs Ours` 对照表生成工具
- 轻量测试、配置系统、云端启动脚本

## 目录结构

```text
configs/                 分层实验配置
data/
  raw/                   可选的数据放置目录
  interim/               缓存、中间产物、抽样结果
  processed/             规范化问题集、语料与索引元数据
src/biorag/              主代码
tests/                   toy fixtures 与 smoke tests
scripts/                 辅助脚本
```

## 安装

最小安装：

```bash
python3 -m pip install -e .
```

云端 / GPU 环境建议安装：

```bash
python3 -m pip install -e '.[cloud]'
```

## 默认实验协议

默认严格协议由 [configs/dataset/bioasq12b_report_strict.yaml](configs/dataset/bioasq12b_report_strict.yaml) 定义，含义如下：

- 训练数据源：`BioASQ-training12b.zip`
- 辅助官方测试数据：`Task12BGoldenEnriched.zip`
- 评测协议：`report_holdout`
- 抽样大小：`500`
- 抽样随机种子：`42`
- 训练 / 评测问题重叠：`0`
- 默认语料模式：`linked_pubmed`
- 文档文本：`title + abstract`
- 生成策略：`greedy decoding`

这里的 `report_holdout` 是对齐报告的实验设定，而不是对齐公开 leaderboard：

- 从 `training12b` 中按题型分层抽取 `500` 条评测题
- 剩余问题用于 retriever 训练
- 同时保留 `golden` 压缩包，方便后续切到官方公开测试口径

如果你想改成官方 `golden` 批次评测，可切换到 [configs/dataset/bioasq12b_golden_eval.yaml](configs/dataset/bioasq12b_golden_eval.yaml)。

## 模型与流水线

默认模型与报告保持一致：

- Retriever: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generator: `Qwen/Qwen2.5-7B-Instruct`

仓库内置两种主模式：

- `baseline`
  - 冻结 bi-encoder
  - 直接 top-3 检索
  - 不走 reranker
- `full_biorag`
  - 先做对比学习微调
  - top-20 检索
  - cross-encoder 重排到 top-3
  - 再进入生成阶段

另外每个模型模块都支持两种加载方式：

- `mode: pretrained`
  - 直接从 `model_name` 加载
- `mode: finetuned`
  - 仍保留相同 `model_name`
  - 但优先从 `checkpoint_path` 加载本地微调权重

## 常用命令

如果官方压缩包在 `data/raw/` 下，可以直接运行：

```bash
biorag prepare-data --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name prep --device cpu
biorag build-corpus --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name corpus --device cpu
biorag build-index --config configs/baseline.yaml --input-dir data/raw --output-dir outputs --run-name baseline-index --device cuda
biorag train-retriever --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name train --device cuda
biorag run-baseline --config configs/baseline.yaml --input-dir data/raw --output-dir outputs --run-name baseline --device cuda
biorag run-full-pipeline --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name full --device cuda
biorag-report --baseline-report outputs/baseline/evaluation/evaluation_report.json --ours-report outputs/full/evaluation/evaluation_report.json --output-dir outputs/comparison
```

默认需要的压缩包文件名是：

- `BioASQ-training12b.zip`
- `Task12BGoldenEnriched.zip`

所有主要命令都支持统一参数：

- `--config`
- `--input-dir`
- `--output-dir`
- `--run-name`
- `--device`

每次运行都会把最终生效配置保存到对应 run 目录，方便复现实验。

## 评测与报告表

当前评测模块会输出：

- Yes/No: `Accuracy`、`F1 Yes`、`F1 No`、`Macro-F1`
- Factoid: `Strict Accuracy`、`Lenient Accuracy`、`MRR`
- List: `Mean Precision`、`Recall`、`F-Measure`
- Overall: `ROUGE-L`、`BERTScore`、`Average Latency`

如果你已经分别跑出了 baseline 和 full pipeline 的评测报告，可以再执行：

```bash
biorag-report --baseline-report outputs/baseline/evaluation/evaluation_report.json --ours-report outputs/full/evaluation/evaluation_report.json --output-dir outputs/comparison
```

它会生成：

- `paper_comparison.md`
- `paper_comparison.json`

格式上尽量贴近报告里的 `Baseline / Ours / Gain` 对照表。

## 当前已经收紧的关键点

- 直接从官方 zip 读取问题集，而不是依赖手动解压后的单 JSON
- PubMed URL 会先规范化为 PMID，再进入语料构建、训练和评测
- 默认切分已经保证训练问题与评测问题不重叠
- 生成模块会复用同一个已加载模型，而不是每题重复加载
- 延迟统计现在覆盖 retrieval、rerank 和 generation
- CLI 在运行前会根据 `runtime.seed` 设置 Python / NumPy / Torch 的随机种子

## 云端运行建议

1. 安装 `.[cloud]`
2. 准备 `BioASQ-training12b.zip` 与 `Task12BGoldenEnriched.zip`
3. 如果用 `linked_pubmed`，设置环境变量 `NCBI_EMAIL`
4. 先执行 `prepare-data`
5. 检查 `outputs/<run>/canonical/split_manifest.json`
6. 再执行 `build-corpus`
7. 然后运行 `run-baseline` 或 `run-full-pipeline`
8. 两套报告都出来后，执行 `biorag-report`

## 说明

这份仓库现在已经更接近“严格协议复现”，但要想让指标数值真正逼近报告表格，仍然需要在云端正式跑训练和推理。也就是说，当前仓库已经把：

- 数据入口
- 切分协议
- 评测口径
- 结果产物结构

都收紧到了报告级别；最后还差真实实验执行本身。
