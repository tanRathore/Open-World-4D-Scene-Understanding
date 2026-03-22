#!/usr/bin/env bash
set -e

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "no nvidia-smi"
fi

python --version
which python

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
