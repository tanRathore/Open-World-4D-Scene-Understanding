#!/usr/bin/env bash
set -e

ENV_NAME="${ENV_NAME:-gs2}"
REPO_ROOT="${REPO_ROOT:-$HOME/Grounded-SAM-2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "need conda"
  exit 1
fi

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -U pip setuptools wheel

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if [ ! -d "$REPO_ROOT" ]; then
  git clone https://github.com/IDEA-Research/Grounded-SAM-2.git "$REPO_ROOT"
fi

cd "$REPO_ROOT"

SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"

echo "ok"
echo "$ENV_NAME"
echo "$REPO_ROOT"
