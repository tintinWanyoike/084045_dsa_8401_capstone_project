#!/usr/bin/env bash
# Runs every notebook in order, top to bottom, saving outputs in place.
# This is the "runnable from scratch" entry point required by the brief.
#
# Usage (from the repository root):
#   bash run_pipeline.sh            # all notebooks
#   bash run_pipeline.sh 02         # only notebooks whose name starts with 02
set -euo pipefail

cd "$(dirname "$0")"
PATTERN="${1:-}"

for nb in notebooks/${PATTERN}*.ipynb; do
    echo "=== Running ${nb}"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 "${nb}"
done
echo "=== Done"
