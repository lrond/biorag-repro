# BioRAG Quickstart

This guide is the shortest path from a fresh clone to a reproducible BioRAG run.
For AutoDL-specific commands, see [docs/AUTODL.md](docs/AUTODL.md).

## 1. Clone And Install

```bash
git clone https://github.com/lrond/biorag-repro.git
cd biorag-repro
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[cloud,dev]'
```

If the machine uses a slow default PyPI mirror, configure your preferred mirror
before the install step.

## 2. Prepare Data

Place the two BioASQ archives here:

```text
data/raw/BioASQ-training12b.zip
data/raw/Task12BGoldenEnriched.zip
```

For the default `linked_pubmed` corpus mode, set an NCBI email:

```bash
export NCBI_EMAIL="you@example.com"
```

## 3. Check The Machine

Run the doctor command before launching a long job:

```bash
biorag doctor --profile full --device cuda
```

Use strict mode in scripts or CI:

```bash
biorag doctor --profile full --device cuda --strict
```

The doctor checks Python, data paths, configured zip members, required packages,
CUDA visibility, and NCBI email configuration.

## 4. Run Baseline And Full BioRAG

```bash
mkdir -p logs
biorag quickstart --profile baseline --device cuda > logs/baseline.log 2>&1
biorag quickstart --profile full --device cuda > logs/full.log 2>&1
```

The default run directories are:

```text
outputs/baseline/
outputs/full/
```

Watch progress with:

```bash
tail -f logs/full.log
```

## 5. Create The Paper-Style Artifacts

Comparison table:

```bash
biorag-report \
  --baseline-report outputs/baseline/evaluation/evaluation_report.json \
  --ours-report outputs/full/evaluation/evaluation_report.json \
  --output-dir outputs/comparison
```

Training convergence curve:

```bash
biorag-plot \
  --training-metrics outputs/full/training/training_metrics.json \
  --output-dir outputs/full/training
```

Expected artifacts:

```text
outputs/comparison/results_comparison.md
outputs/full/training/training_loss_curve.csv
outputs/full/training/training_loss_curve.json
outputs/full/training/training_loss_curve.png
```

## 6. Common Runtime Notes

- The first run can spend a long time fetching PubMed records. Current code
  caches every resolved PMID under `data/interim/pubmed_cache/`, so restarted
  runs reuse completed fetches.
- `baseline` skips retriever fine-tuning and cross-encoder reranking.
- `full` trains the retriever, rebuilds the FAISS index with the trained model,
  reranks top-20 candidates, sends top-3 documents to Qwen2.5-7B-Instruct, and
  evaluates the generated answers.
- If GPU memory is tight, lower `inference.max_prompt_characters` or
  `models.generator.max_new_tokens` in the active config.
