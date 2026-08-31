# Training env (DSNet / CLIP / eval).  Use:  source env/activate-train.sh
export PVS_SCRATCH=/var/tmp/akn57
export PIP_CACHE_DIR=$PVS_SCRATCH/cache/pip
export HF_HOME=$PVS_SCRATCH/cache/huggingface
export TORCH_HOME=$PVS_SCRATCH/cache/torch
source "$PVS_SCRATCH/envs/pvs-train/bin/activate"
echo "pvs-train active: $(python -V 2>&1), torch $(python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
echo "NOTE: shared GPUs - set CUDA_VISIBLE_DEVICES before training (e.g. export CUDA_VISIBLE_DEVICES=0)"
