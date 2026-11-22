#!/usr/bin/env bash
# Runs the whole pipeline notebook top to bottom, headless, saving outputs in place.
# This is the "runnable from scratch" entry point required by the brief.
#
# Usage (from the repository root):
#   bash run_pipeline.sh         

set -euo pipefail

cd "$(dirname "$0")"
echo "=== Running notebook.ipynb"
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebook.ipynb
echo "=== Done. Outputs: data/, models/, reports/ (figures in reports/figures/)"
