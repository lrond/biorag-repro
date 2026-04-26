# Report Alignment Notes

This project is structured to match the BioRAG report pipeline while keeping the
implementation reproducible and easier to run on cloud GPUs.

## Architecture Checklist

| Report item | Project implementation |
| --- | --- |
| BioASQ 12b data | `configs/dataset/bioasq12b_report_strict.yaml` |
| 500-query stratified holdout | `report_holdout` protocol in `src/biorag/datasets.py` |
| No train/eval question overlap | `train_questions.jsonl` excludes held-out IDs |
| Baseline frozen PubMedBERT bi-encoder | `configs/baseline.yaml` |
| Baseline top-3 retrieval | `run_baseline` forces `retrieve_top_k=3` |
| Full retriever InfoNCE fine-tuning | `train_contrastive_retriever` |
| Temperature 0.05 | `configs/training/full_biorag.yaml` |
| 3 epochs, batch size 16, LR 2e-5 | `configs/training/full_biorag.yaml` |
| Top-20 retrieval before rerank | `configs/full_biorag.yaml` |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Top-3 context to generator | `inference.final_top_k=3` |
| Qwen2.5-7B-Instruct generator | `configs/models/full_biorag.yaml` |
| Greedy decoding | `do_sample=false`, `temperature=0.0` |
| Training convergence curve | `biorag-plot` |
| Baseline/Ours/Gain table | `biorag-report` |

## Important Differences From The Original Report Folder

The `report_version/` folder is treated as a local reference artifact, not the
source of truth for this repository. The production project improves on it in a
few ways:

- It uses explicit config files instead of manually editing model paths.
- It materializes train/evaluation splits and records the sampled IDs.
- It avoids train/eval question overlap for the report-style protocol.
- It writes PubMed fetches incrementally to the cache.
- It uses the saved FAISS index for retrieval when FAISS is available.
- It records step-level retriever losses for convergence plots.
- It has tests for model mode resolution, dataset splits, metrics, plotting,
  retrieval contracts, and CLI entry points.

## Reproduce The Report-Style Run

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

The exact table values can still vary with model library versions, GPU kernels,
and upstream model revisions. For locked-down reproduction, pin dependency
versions and cache the Hugging Face model snapshots used for the experiment.

## Known Experimental Boundary

The default corpus mode is `linked_pubmed`: it builds a corpus from the PubMed
articles linked by BioASQ question metadata. This matches the report-style
project scope and keeps the experiment tractable, but it is not a full PubMed
production-scale search index. Use `configs/dataset/bioasq12b_pubmed_dump.yaml`
when you have a larger local PubMed dump.
