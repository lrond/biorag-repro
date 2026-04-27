# BioRAG

BioRAG 是一个面向 BioASQ 生物医学问答任务的检索增强生成项目。项目包含
数据准备、PubMed 语料构建、FAISS 检索索引、PubMedBERT 检索器微调、
Cross-Encoder 重排、Qwen2.5 生成、BioASQ-style 评估、错误分析和训练曲线绘制。

## 项目结构

```text
src/
  check_setup.py               # 环境、数据和 GPU 自检
  prepare_data.py              # 从 BioASQ zip 构建 train/eval/all processed jsonl
  train_retriever.py           # PubMedBERT + MultipleNegativesRankingLoss
  build_index.py               # 构建 BM25 + FAISS index
  retrieval.py                 # 检索器封装
  reranker.py                  # Cross-encoder reranker
  rag_engine_baseline.py       # baseline: PubMedBERT top-3 + Qwen
  rag_engine.py                # full BioRAG: top-20 + rerank top-3 + Qwen
  batch_inference_baseline.py  # baseline 批量推理
  batch_inference.py           # full BioRAG 批量推理
  evaluate_official.py         # BioASQ-style 指标与对照表
  plot_training_curve.py       # 训练收敛曲线
  analyze_errors.py            # 错误分析
  test_latency.py              # 延迟测试
  run_pipeline.py              # 一键串联流程，支持 resume/force
data/
  raw/                         # BioASQ 原始 zip
  processed/                   # 生成的 processed jsonl
  interim/pubmed_cache/        # PubMed 增量缓存
  indexes/                     # FAISS/BM25 index
models/                        # 微调模型
outputs/                       # 预测、评估、曲线
presentation/                  # 展示 slides、Beamer 源码和讲稿
```

## 环境安装

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

AutoDL 或国内网络可以先设置镜像：

```bash
python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

如果要从 NCBI 拉 PubMed 摘要，建议设置邮箱：

```bash
export NCBI_EMAIL="you@example.com"
```

仓库中的 `.env.example` 只作为环境变量清单，运行脚本前请用 `export` 设置需要的变量。

安装后先做一次自检：

```bash
python src/check_setup.py --device cuda
```

如果希望脚本在发现缺依赖或没有 GPU 时直接失败：

```bash
python src/check_setup.py --device cuda --strict
```

## 数据

仓库默认使用：

```text
data/raw/BioASQ-training12b.zip
data/raw/Task12BGoldenEnriched.zip
```

`prepare_data.py` 会从 training zip 中做 500 条分层 holdout，并保证训练问题和
评测问题不重叠。PubMed 摘要会按 PMID 增量缓存到
`data/interim/pubmed_cache/`，中断后重跑会复用已经抓到的数据。

## 语料范围

本项目使用 **BioASQ-linked PubMed evidence corpus**：系统根据 BioASQ
问题、documents 和 snippets 中引用的 PMID 收集对应的 PubMed 摘要，并在这些
文档上构建检索语料。这样可以在课程级算力限制下保证实验可复现，也能让完整
pipeline 在单张云端 GPU 上跑通。

本项目没有索引完整 PubMed 数据库。完整 PubMed-scale 部署可以作为后续扩展，
需要本地 PubMed dump、大规模 embedding 生成、IVF/HNSW 等近似最近邻索引、
索引分片，以及更大的存储和计算资源。

## 一键运行

先跑 baseline：

```bash
mkdir -p logs
python src/run_pipeline.py --profile baseline --device cuda > logs/baseline.log 2>&1
```

再跑 full BioRAG：

```bash
python src/run_pipeline.py --profile full --device cuda > logs/full.log 2>&1
```

查看进度：

```bash
tail -f logs/full.log
```

如果想一次跑完 baseline 和 full：

```bash
python src/run_pipeline.py --profile all --device cuda
```

默认情况下，`run_pipeline.py` 会自动跳过已经完成的阶段，方便中断后继续：

```bash
python src/run_pipeline.py --profile full --device cuda --resume
```

如果需要强制重跑全部阶段：

```bash
python src/run_pipeline.py --profile full --device cuda --force
```

如果只是调试脚本、暂时不想检查 GPU 和依赖：

```bash
python src/run_pipeline.py --profile baseline --device cpu --skip-check
```

## 分阶段运行

### 1. 准备数据

```bash
python src/prepare_data.py
```

主要输出：

```text
data/processed/train_processed.jsonl
data/processed/eval_processed.jsonl
data/processed/all_processed.jsonl
data/processed/split_manifest.json
```

### 2. Baseline

```bash
python src/build_index.py --model base --device cuda
python src/batch_inference_baseline.py --device cuda
python src/evaluate_official.py
```

Baseline 对齐报告里的设置：PubMedBERT-base frozen bi-encoder，FAISS dense
top-3，直接送入 Qwen2.5-7B-Instruct。

### 3. Full BioRAG

```bash
python src/train_retriever.py --device cuda
python src/build_index.py --model finetuned --device cuda
python src/batch_inference.py --device cuda
python src/evaluate_official.py
```

Full BioRAG 对齐报告里的设置：InfoNCE/MultipleNegativesRankingLoss 微调
retriever，先检索 top-20，再用 `cross-encoder/ms-marco-MiniLM-L-6-v2`
重排并选 top-3，最后交给 Qwen2.5-7B-Instruct 生成。

### 4. 曲线和分析

```bash
python src/plot_training_curve.py
python src/analyze_errors.py
python src/test_latency.py --mode baseline --device cuda
python src/test_latency.py --mode full --device cuda
```

## 输出文件

主要输出集中在 `outputs/`：

```text
outputs/predictions_baseline.json
outputs/predictions.json
outputs/evaluation_results.md
outputs/evaluation_results.json
outputs/training_metrics.json
outputs/training_loss_curve.csv
outputs/training_loss_curve.png
outputs/error_analysis.json
outputs/run_manifest.json
```

`run_manifest.json` 会记录本次运行的 profile、device、resume/force 参数和各阶段
命令，便于复现实验和排查中断点。

## 展示材料

展示文件放在 `presentation/`：

```text
presentation/slides_en.pdf
presentation/slides_zh.pdf
presentation/slides_zh.tex
presentation/speaker_notes.pdf
presentation/aligned_training_loss.png
```

如果需要重新编译中文 Beamer：

```bash
cd presentation
xelatex slides_zh.tex
```

## 关键超参

默认超参集中在 `src/config.py`：

| 项目 | 默认值 |
| --- | --- |
| Retriever base | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `Qwen/Qwen2.5-7B-Instruct` |
| Holdout | 500 条分层随机样本，seed=42 |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| InfoNCE temperature | 0.05 |
| Baseline retrieval | top-3 |
| Full retrieval | top-20 -> rerank top-3 |
| Generation | greedy decoding |

常用环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export NCBI_EMAIL=you@example.com
export BIORAG_USE_BM25=0
```

`BIORAG_USE_BM25=0` 是默认值，使用 FAISS dense retrieval。如果想启用
BM25 + dense 的混合检索，可以设置为 `1`。

## AutoDL 常用命令

```bash
cd /root/autodl-tmp
git clone git@github.com:lrond/biorag-repro.git nlp
cd nlp
python -m pip install -r requirements.txt
export NCBI_EMAIL="you@example.com"
python src/check_setup.py --device cuda
mkdir -p logs
screen -S biorag
python src/run_pipeline.py --profile all --device cuda > logs/run.log 2>&1
```

退出 screen：按 `Ctrl-A` 再按 `D`。

监控：

```bash
tail -f logs/run.log
find data/interim/pubmed_cache -name '*.json' | wc -l
nvidia-smi
```

停止：

```bash
pkill -f "src/run_pipeline.py"
```
