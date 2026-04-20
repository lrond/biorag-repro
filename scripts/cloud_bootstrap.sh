#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -e '.[cloud]'

echo "BioRAG cloud environment is ready."
echo "Next suggested steps:"
echo "  1. Place BioASQ files in data/raw/"
echo "  2. Export NCBI_EMAIL if using linked_pubmed"
echo "  3. Run: biorag run-full-pipeline --config configs/full_biorag.yaml --input-dir data/raw --output-dir outputs --run-name full --device cuda"
