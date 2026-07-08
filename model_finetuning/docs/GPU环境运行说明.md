# GPU 环境运行说明

当前已创建可用环境：

```bash
/data1/shared_data/qwen3vl-showeeData/.venv
```

已验证 `torch.cuda.is_available()` 为 `True`，可见 8 张 NVIDIA A800-SXM4-80GB。

## 1. 进入项目目录

```bash
cd /data1/shared_data/qwen3vl-showeeData
nvidia-smi
```

确认能看到 GPU 后继续。

## 2. 创建项目 Python 环境

```bash
bash scripts/setup_qwen3vl_env.sh
```

该脚本会创建：

```bash
/data1/shared_data/qwen3vl-showeeData/.venv
```

并安装：

- PyTorch CUDA 12.6 wheel
- torchvision CUDA 12.6 wheel
- transformers 5.12.1
- accelerate
- safetensors
- pillow
- decord
- av
- opencv-python
- tqdm

当前视频 smoke 脚本使用 `decord` 预抽 16 帧后传给 processor，避免 `transformers` 默认视频解码路径依赖 `torchcodec` 或已弃用的 `torchvision.io.read_video`。

注意：当前系统 Python 缺少 `ensurepip`，直接 `python3 -m venv` 会失败。安装脚本已优先使用 `virtualenv`；如果新机器没有 `virtualenv`，先执行：

```bash
python3 -m pip install --user --upgrade virtualenv -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 跑视频推理 smoke test

```bash
bash scripts/run_video_smoke.sh
```

输入数据：

```bash
data/processed/smoke_train.json
```

模型路径：

```bash
models/base/Qwen3-VL-8B-Instruct
```

输出位置：

```bash
runs/puzuo/video_smoke_infer/output.json
runs/puzuo/video_smoke_infer/run.log
runs/puzuo/video_smoke_infer/nvidia-smi.log
```

## 4. 如果失败，优先检查

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; print('ok')"
```

如果 `torch.cuda.is_available()` 是 `False`，说明当前 shell 虽然可能有 CUDA 工具，但 Python 环境仍没有正确访问 GPU。
