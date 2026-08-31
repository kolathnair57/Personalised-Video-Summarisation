# Inference env (vLLM / Qwen3-VL teacher).  Use:  source env/activate-vllm.sh
export PVS_SCRATCH=/var/tmp/akn57
export PIP_CACHE_DIR=$PVS_SCRATCH/cache/pip
export HF_HOME=$PVS_SCRATCH/cache/huggingface
export TORCH_HOME=$PVS_SCRATCH/cache/torch
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # needed for multimodal (runbook 5.1)
source "$PVS_SCRATCH/envs/pvs-vllm/bin/activate"
echo "pvs-vllm active: vllm $(vllm --version 2>/dev/null)"
echo "NOTE: RTX 3090 = sm_86, NO FP8. Use bf16 8B; do NOT pull *-FP8 checkpoints."
echo "NOTE: do NOT upgrade vllm past 0.19.1 - 0.20.0+ needs CUDA 13 (driver >= 580, this box has 555)."
