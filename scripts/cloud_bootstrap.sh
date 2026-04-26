#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -e '.[cloud,dev]'

echo "BioRAG cloud environment is ready."
echo "Next suggested steps:"
echo "  1. Place BioASQ files in data/raw/"
echo "  2. Export NCBI_EMAIL if using linked_pubmed"
echo "  3. Run: biorag doctor --profile full --device cuda"
echo "  4. Run: biorag quickstart --profile baseline --device cuda"
echo "  5. Run: biorag quickstart --profile full --device cuda"
