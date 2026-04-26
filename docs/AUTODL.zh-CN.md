# AutoDL 运行手册

这份手册假设你已经有一个带 CUDA GPU 的 AutoDL 容器。

## 1. 登录并克隆

```bash
ssh -p <port> root@<autodl-host>
cd /root/autodl-tmp
git clone git@github.com:lrond/biorag-repro.git nlp
cd nlp
```

如果 SSH clone 不方便，可以使用带仓库权限 token 的 HTTPS。

## 2. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[cloud,dev]'
```

可选：切到国内镜像。

```bash
python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

## 3. 检查数据和 GPU

确认有：

```text
data/raw/BioASQ-training12b.zip
data/raw/Task12BGoldenEnriched.zip
```

然后运行：

```bash
export NCBI_EMAIL="you@example.com"
biorag doctor --profile full --device cuda
nvidia-smi
```

## 4. 用 screen 后台跑

```bash
mkdir -p logs
screen -S biorag
biorag quickstart --profile baseline --device cuda > logs/baseline.log 2>&1
```

按 `Ctrl-A` 再按 `D` 可以退出 screen 但不中断任务。

baseline 完成后跑 full：

```bash
screen -S biorag-full
biorag quickstart --profile full --device cuda > logs/full.log 2>&1
```

## 5. 监控和停止

```bash
tail -f logs/full.log
ps aux | grep biorag | grep -v grep
find data/interim/pubmed_cache -name '*.json' | wc -l
nvidia-smi
```

如果在 screen 里，按 `Ctrl-C` 停止当前任务。另一个 shell 里可以用：

```bash
pkill -f "biorag quickstart"
```

PubMed 缓存是增量写入的，所以停止后重跑不会丢掉已经抓到的 PMID。

## 6. 生成表格和曲线

```bash
biorag-report \
  --baseline-report outputs/baseline/evaluation/evaluation_report.json \
  --ours-report outputs/full/evaluation/evaluation_report.json \
  --output-dir outputs/comparison

biorag-plot \
  --training-metrics outputs/full/training/training_metrics.json \
  --output-dir outputs/full/training
```

## 7. 用 HTTP 端口预览输出

如果 AutoDL 给了 `6006` 或 `6008` 这类 HTTP 服务端口，可以这样开：

```bash
python -m http.server 6008 -d outputs
```

然后在 AutoDL 控制台打开对应的自定义服务地址。

## 8. 更新到最新 main

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
python -m pip install -e '.[cloud,dev]'
biorag doctor --profile full --device cuda
```
