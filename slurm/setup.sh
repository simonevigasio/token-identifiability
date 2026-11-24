#!/usr/bin/env bash
# One-time setup on AAU AI-LAB / AI Cloud. From the repository root:
#
#   srun --mem=24G --cpus-per-task=8 --gres=gpu:1 --time=01:00:00 bash slurm/setup.sh
#
# Needs internet (pip, GitHub, Hugging Face). If the compute node has none, run it on
# the front-end node instead: `bash slurm/setup.sh` (it is light: ~3 GB of downloads).
#
# 1. venv on top of the AAU PyTorch container (torch/CUDA come from the container)
# 2. pip install the remaining packages
# 3. official ELF PyTorch code at the pinned commit -> .deps/elf-pytorch
# 4. download T5-small, T5-base, ELF-B (OWT, WMT14 De-En, XSum), GLUE, WMT14 and the
#    docmix / multidoc prompts (XSum validation + Pile test split, ~0.4 GB) into $HF_HOME
# 5. unit tests + GPU check
set -euo pipefail
source "$(dirname "$0")/common.sh"
DATASETS=${DATASETS:-"mrpc cola mnli wmt14 docmix multidoc"}
echo "container: $CONTAINER"
echo "venv:      $VENV"
echo "HF_HOME:   $HF_HOME"

if [ ! -x "$VENV/bin/python" ]; then
  "${CONTAINER_CMD[@]}" python3 -m venv --system-site-packages "$VENV"
  echo "$CONTAINER" > "$VENV/.container"
fi
# Do NOT reinstall torch: the container's build matches the cluster's CUDA driver.
in_container python -m pip install "transformers>=4.45,<5" "huggingface-hub>=0.25,<1" "datasets>=2.19" \
  "sentencepiece>=0.2" "einops>=0.7" matplotlib PyYAML pytest zstandard

if [ ! -d .deps/elf-pytorch ]; then
  in_container python scripts/prepare_elf.py || python3 scripts/prepare_elf.py
fi

# shellcheck disable=SC2086
in_container python -m tokid.prefetch --models t5-small t5-base elf-b elf-b-de-en elf-b-xsum --datasets $DATASETS
in_container python -m pytest -q
in_container python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', \
transformers.__version__, '| CUDA available:', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
mkdir -p logs
echo "Setup complete."
