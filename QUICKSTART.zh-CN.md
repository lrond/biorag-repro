# BioRAG 快速开始

这份文档适合从零克隆项目后直接跑实验。AutoDL 专用操作见
[docs/AUTODL.zh-CN.md](docs/AUTODL.zh-CN.md)。

## 1. 克隆与安装

```bash
git clone https://github.com/lrond/biorag-repro.git
cd biorag-repro
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[cloud,dev]'
```

如果下载依赖很慢，可以先切换到国内 PyPI 镜像。

## 2. 准备数据

把两个 BioASQ 压缩包放在：

```text
data/raw/BioASQ-training12b.zip
data/raw/Task12BGoldenEnriched.zip
```

默认 `linked_pubmed` 模式会从 NCBI 拉取 PubMed 摘要，建议设置邮箱：

```bash
export NCBI_EMAIL="you@example.com"
```

## 3. 先做自检

长时间任务开始前先跑：

```bash
biorag doctor --profile full --device cuda
```

脚本化执行时可以用严格模式：

```bash
biorag doctor --profile full --device cuda --strict
```

它会检查 Python 版本、数据路径、zip 内部成员、依赖包、CUDA 可见性和
NCBI 邮箱配置。

## 4. 跑 Baseline 和 Full BioRAG

```bash
mkdir -p logs
biorag quickstart --profile baseline --device cuda > logs/baseline.log 2>&1
biorag quickstart --profile full --device cuda > logs/full.log 2>&1
```

默认输出目录：

```text
outputs/baseline/
outputs/full/
```

查看进度：

```bash
tail -f logs/full.log
```

## 5. 生成论文风格产物

对照表：

```bash
biorag-report \
  --baseline-report outputs/baseline/evaluation/evaluation_report.json \
  --ours-report outputs/full/evaluation/evaluation_report.json \
  --output-dir outputs/comparison
```

训练收敛曲线：

```bash
biorag-plot \
  --training-metrics outputs/full/training/training_metrics.json \
  --output-dir outputs/full/training
```

常见输出：

```text
outputs/comparison/results_comparison.md
outputs/full/training/training_loss_curve.csv
outputs/full/training/training_loss_curve.json
outputs/full/training/training_loss_curve.png
```

## 6. 运行时说明

- 第一次跑可能会花很多时间抓 PubMed。现在每个 PMID 会增量缓存到
  `data/interim/pubmed_cache/`，中断后重跑会复用已有缓存。
- `baseline` 不训练 retriever，也不跑 cross-encoder rerank。
- `full` 会训练 retriever，用训练后的模型重建 FAISS index，检索 top-20，
  rerank 后取 top-3 给 Qwen2.5-7B-Instruct，并输出评测。
- 如果显存紧张，优先降低配置里的 `inference.max_prompt_characters` 或
  `models.generator.max_new_tokens`。
