# 论文/报告对齐说明

当前项目把原报告里的 BioRAG 流程整理成可复现的工程版本，同时保留了按阶段
运行和排查中间产物的能力。

## 架构对齐清单

| 论文/报告描述 | 当前项目实现 |
| --- | --- |
| BioASQ 12b 数据 | `configs/dataset/bioasq12b_report_strict.yaml` |
| 500 条分层 holdout | `src/biorag/datasets.py` 的 `report_holdout` 协议 |
| train/eval 问题无重叠 | `train_questions.jsonl` 排除 held-out IDs |
| Baseline 冻结 PubMedBERT bi-encoder | `configs/baseline.yaml` |
| Baseline top-3 retrieval | `run_baseline` 强制 `retrieve_top_k=3` |
| Full retriever InfoNCE 微调 | `train_contrastive_retriever` |
| temperature=0.05 | `configs/training/full_biorag.yaml` |
| 3 epochs, batch size 16, LR 2e-5 | `configs/training/full_biorag.yaml` |
| rerank 前 top-20 检索 | `configs/full_biorag.yaml` |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| top-3 context 给生成器 | `inference.final_top_k=3` |
| Qwen2.5-7B-Instruct | `configs/models/full_biorag.yaml` |
| greedy decoding | `do_sample=false`, `temperature=0.0` |
| 训练收敛曲线 | `biorag-plot` |
| Baseline/Ours/Gain 表格 | `biorag-report` |

## 相比 report_version 的改进

`report_version/` 只是本地参考材料，不作为正式项目源代码。当前工程版做了这些
整理：

- 用配置文件切换模型，不需要手动改脚本变量。
- 真实落盘 train/eval split，并记录抽样 ID。
- `report_holdout` 协议保证训练问题和评测问题无重叠。
- PubMed fetch 增量缓存，中断后可继续。
- 有 FAISS 时实际使用持久化索引检索。
- 训练记录 step-level loss，可直接画收敛曲线。
- 增加了数据协议、模型模式、检索、评估、绘图、CLI 等测试。

## 复现报告风格实验

```bash
biorag doctor --profile full --device cuda
biorag quickstart --profile baseline --device cuda
biorag quickstart --profile full --device cuda
biorag-report \
  --baseline-report outputs/baseline/evaluation/evaluation_report.json \
  --ours-report outputs/full/evaluation/evaluation_report.json \
  --output-dir outputs/comparison
biorag-plot \
  --training-metrics outputs/full/training/training_metrics.json \
  --output-dir outputs/full/training
```

精确数值仍可能受依赖版本、GPU kernel、Hugging Face 模型快照影响。如果要锁死
复现实验，需要固定依赖版本并缓存当时的模型 snapshot。

## 实验边界

默认 `linked_pubmed` 语料模式会使用 BioASQ 问题元数据关联到的 PubMed 文章。
这和报告项目规模一致，也适合课程/云端复现；但它不是完整 PubMed 生产级索引。
如果你有更大的本地 PubMed dump，可以改用
`configs/dataset/bioasq12b_pubmed_dump.yaml`。
