#!/usr/bin/env bash
set -e

REPO_DIR="${REPO_DIR:-/workspace/Grounded-SAM-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_DIR="$SCRIPT_DIR"

cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
echo "repo: $REPO_DIR"
echo "job:  $JOB_DIR"

mkdir -p "$JOB_DIR/raw_grounded_sam2"

$PYTHON_BIN "$JOB_DIR/run_grounded_sam2_hf.py"

echo "done"
echo "$JOB_DIR/preds.json"