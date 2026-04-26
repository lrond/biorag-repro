# BioRAG Report Version

这是基于 `report_version/team-2` 重新整理后的项目版本。旧的工程化 package、
多份 quickstart 和分散文档已经移除；当前 `main` 只保留报告版风格的脚本流程，
并修正了原 report version 里的路径、切分、检索和评估问题。

## 项目结构

```text
src/
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
  run_pipeline.py              # 一键串联流程
data/
  raw/                         # BioASQ 原始 zip
  processed/                   # 生成的 processed jsonl
  interim/pubmed_cache/        # PubMed 增量缓存
  indexes/                     # FAISS/BM25 index
models/                        # 微调模型
outputs/                       # 预测、评估、曲线
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

## 数据

仓库默认使用：

```text
data/raw/BioASQ-training12b.zip
data/raw/Task12BGoldenEnriched.zip
```

`prepare_data.py` 会从 training zip 中做 500 条分层 holdout，并保证训练问题和
评测问题不重叠。PubMed 摘要会按 PMID 增量缓存到
`data/interim/pubmed_cache/`，中断后重跑会复用已经抓到的数据。

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

`BIORAG_USE_BM25=0` 是默认值，更贴近论文里的 FAISS dense retrieval。
如果想复用原 report version 的混合检索思路，可以设置为 `1`。

## AutoDL 常用命令

```bash
cd /root/autodl-tmp
git clone git@github.com:lrond/biorag-repro.git nlp
cd nlp
python -m pip install -r requirements.txt
export NCBI_EMAIL="you@example.com"
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

## 这版相比原 report_version 修了什么

- 不再把核心代码藏在 `.ipynb_checkpoints/`。
- 不需要手动改 `MODEL_NAME`，`build_index.py --model base|finetuned` 直接切换。
- 不再用前 500 条假装 holdout，而是 500 条分层随机 holdout。
- 训练集和评测集按 question id 去重，避免问题级重叠。
- PubMed 缓存增量写入，长时间 fetch 中断后可继续。
- 修复原 `retrieval.py` 在 dense 循环内提前 `return` 的问题。
- 默认使用 FAISS dense retrieval，更贴近论文；BM25 只作为可选开关。
- 评估增加 MRR、list precision/recall/F1、ROUGE-L、BERTScore 和对照表输出。
- 训练保存 step-level loss，可直接画收敛曲线。
