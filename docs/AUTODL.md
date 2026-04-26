# AutoDL Runbook

This runbook assumes a fresh AutoDL container with a CUDA GPU.

## 1. Connect And Clone

```bash
ssh -p <port> root@<autodl-host>
cd /root/autodl-tmp
git clone git@github.com:lrond/biorag-repro.git nlp
cd nlp
```

If SSH is unavailable, HTTPS also works when the token has repository access:

```bash
git clone https://github.com/lrond/biorag-repro.git nlp
```

## 2. Install

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[cloud,dev]'
```

Optional mirror example:

```bash
python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

## 3. Check Data And GPU

The repository expects:

```text
data/raw/BioASQ-training12b.zip
data/raw/Task12BGoldenEnriched.zip
```

Then run:

```bash
export NCBI_EMAIL="you@example.com"
biorag doctor --profile full --device cuda
nvidia-smi
```

## 4. Run In A Detachable Session

```bash
mkdir -p logs
screen -S biorag
biorag quickstart --profile baseline --device cuda > logs/baseline.log 2>&1
```

Detach from `screen` with `Ctrl-A` then `D`.

Start the full pipeline after the baseline:

```bash
screen -S biorag-full
biorag quickstart --profile full --device cuda > logs/full.log 2>&1
```

## 5. Monitor Or Stop

```bash
tail -f logs/full.log
ps aux | grep biorag | grep -v grep
find data/interim/pubmed_cache -name '*.json' | wc -l
nvidia-smi
```

To stop the active foreground process, press `Ctrl-C` inside the screen session.
To stop it from another shell:

```bash
pkill -f "biorag quickstart"
```

The PubMed cache is incremental, so restarting continues from already cached
PMIDs instead of throwing away all previous fetch work.

## 6. Build Tables And Curves

```bash
biorag-report \
  --baseline-report outputs/baseline/evaluation/evaluation_report.json \
  --ours-report outputs/full/evaluation/evaluation_report.json \
  --output-dir outputs/comparison

biorag-plot \
  --training-metrics outputs/full/training/training_metrics.json \
  --output-dir outputs/full/training
```

## 7. Simple HTTP Preview

If AutoDL exposes HTTP ports such as `6006` or `6008`, you can serve outputs:

```bash
python -m http.server 6008 -d outputs
```

Then open the AutoDL custom service URL for port `6008`.

## 8. Update To The Latest Main

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
python -m pip install -e '.[cloud,dev]'
biorag doctor --profile full --device cuda
```
