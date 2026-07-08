#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data1/shared_data/qwen3vl-showeeData
VENV_DIR="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"

if python3 -m virtualenv --version >/dev/null 2>&1; then
  python3 -m virtualenv "$VENV_DIR"
else
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/
PYPI_TRUSTED_HOST=mirrors.aliyun.com

python -m pip install --upgrade pip setuptools wheel \
  -i "$PYPI_MIRROR" --trusted-host "$PYPI_TRUSTED_HOST"

# CUDA 12.8 driver can run CUDA 12.6 PyTorch wheels. If your server has a
# different policy, replace the index URL with the site-specific mirror.
pip install -i "$PYPI_MIRROR" --trusted-host "$PYPI_TRUSTED_HOST" \
  --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision

pip install -i "$PYPI_MIRROR" --trusted-host "$PYPI_TRUSTED_HOST" \
  transformers accelerate safetensors pillow decord av opencv-python tqdm

python - <<'PY'
import torch
import torchvision
import transformers
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('transformers', transformers.__version__)
print('qwen3vl import ok')
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu0', torch.cuda.get_device_name(0))
PY
