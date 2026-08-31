# Phase 1 — Environment notes

Date set up: 2026-08-11. Machine: `nitt`.

This file records how the actual environment differs from the Build Runbook, which
was written for the HEX cluster. Every deviation below is forced by the hardware or
storage on this machine, not a preference.

## 1. The machine is not HEX

| Runbook assumes | Reality on `nitt` |
|---|---|
| HEX cluster, SLURM batch jobs | Single shared box, **no scheduler** (`sbatch`/`squeue` absent) |
| `module load cuda/12.4` | **No module system.** CUDA 12.5 already at `/usr/local/cuda-12.5` |
| A100/H100, 80 GB, FP8 capable | **3 x RTX 3090**, 24 GB each, Ampere `sm_86` |
| Storage quota on request | Home quota **25 GB**, hard-capped |

Consequences:

- **No pre-emption** (there is no queue), so the runbook's "checkpoint the teacher job
  against pre-emption" is less critical — but `labels_cache/` is still worth it, because
  the box is shared and someone else's job can OOM the GPU you are using.
- **You share GPUs live with other users.** Always set `CUDA_VISIBLE_DEVICES` and never
  take all three cards.

## 2. FP8 is impossible on this hardware

vLLM reports directly:

```
compute cap  : DeviceCapability(major=8, minor=6)
supports fp8 : False
```

FP8 requires Hopper/Ada (`sm_89`+). Ampere `sm_86` has no FP8 tensor cores.

**Phase 5 impact:** `Qwen/Qwen3-VL-32B-Instruct-FP8` **cannot run here at all.**

- Workhorse teacher: `Qwen3-VL-8B-Instruct` in bf16 (~16 GB) fits one 3090 (24 GB).
- The 32B ablation (Phase 11.1) needs a different route: a 4-bit AWQ/GPTQ 32B across
  2 GPUs with `--tensor-parallel-size 2`. Not yet tested.

## 3. Storage layout

Home is quota-capped at 25 GB — far too small for two envs (~17 GB) plus model
weights (~16 GB per model). `/mnt/fast0` (the box's usual scratch, and where
`.bashrc` used to point) has **0 bytes free**; `/mnt/faster0` has only ~42 GB at 98%
full and is shared. So everything heavy lives on the local disk:

```
/var/tmp/akn57/                     # local disk (sda2), ~190 GB free, NOT on your quota
├── envs/pvs-train/                 # 5.5 GB  training env
├── envs/pvs-vllm/                  # 11 GB   inference env
├── cache/{pip,huggingface,torch}/  # was pointed at the full /mnt/fast0
├── old-project/                    # 7.7 GB  copy of the deleted ~/Personalised-Video-Summary
└── backup/
    ├── pvs-old-project_BACKUP.tar.gz        # 237 MB, irreplaceable work only
    ├── pvs-old-project_BACKUP.tar.gz.sha256
    ├── MANIFEST.txt
    └── patches/                             # DSNet + PGL-SUM local modifications as .patch
```

**Caveat:** `/var/tmp` is local to `nitt`. It does not follow you to another machine, and
it is not backed up. `/var/tmp` has no auto-delete rule on this box (only `/tmp` does,
30 days) — but the envs are rebuildable from the lockfiles here, so a loss costs one
rebuild, not any work.

`.bashrc` was updated (backup in `backup/`) to point `HF_HOME`, `TORCH_HOME` and
`PIP_CACHE_DIR` at this scratch instead of the full `/mnt/fast0`.

## 4. Version choices and why

### Training env (`pvs-train`) — torch 2.6.0+cu124

Driver 555.42.06 supports CUDA **12.5**. PyTorch channels offered:

| Channel | Newest for py3.12 | Verdict |
|---|---|---|
| cu124 | **2.6.0** stable | chosen — 12.4 <= 12.5, guaranteed |
| cu126 | only `2.13.0.dev` nightlies | rejected: never pin a dissertation to a nightly |
| cu128 | 2.11.0 stable | rejected here: needs forward-compat, and 2.6 is ample for DSNet |

### Inference env (`pvs-vllm`) — vllm 0.19.1, torch 2.10.0+cu128

vLLM moved to the **CUDA 13** runtime at version 0.20.0. CUDA 13 requires driver >= 580;
this box has 555. Resolved without downloading via `pip install --dry-run --report`:

| vLLM | torch | cuda-runtime | usable here |
|---|---|---|---|
| 0.11.0 – 0.19.1 | 2.8.0 – 2.10.0 | 12.8.90 (cu12) | yes |
| 0.20.0+ | 2.11.0+ | 13.0.96 (cu13) | **no — driver too old** |

**0.19.1 is therefore the newest usable vLLM**, and is well above the runbook's 0.11.0
minimum for Qwen3-VL. Do not upgrade past 0.19.1 unless the driver is upgraded first.

cu128 on a cu125 driver relies on CUDA minor-version compatibility; verified empirically
(bf16 matmul on GPU succeeds), not assumed.

## 5. Reproducing these environments

```bash
# training env
python3 -m venv /var/tmp/akn57/envs/pvs-train
source /var/tmp/akn57/envs/pvs-train/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -c env/constraints-train.txt h5py scipy scikit-learn tqdm pyyaml ortools open_clip_torch ftfy regex

# inference env
python3 -m venv /var/tmp/akn57/envs/pvs-vllm
source /var/tmp/akn57/envs/pvs-vllm/bin/activate
pip install vllm==0.19.1 qwen-vl-utils==0.0.14 openai pillow
```

Exact pins: `requirements-train.txt` (58 pkgs), `requirements-vllm.txt` (181 pkgs).
The `constraints-*.txt` files pin torch so a later `pip install` cannot silently swap in
a PyPI torch built for the wrong CUDA — a failure mode that is very hard to diagnose.

## 6. Checkpoint results (Phase 1, passed)

```
train env : torch 2.6.0+cu124, cuda available True, 3x RTX 3090 sm_86,
            GPU matmul OK, ortools knapsack solves, all 12 imports OK
vllm env  : vllm 0.19.1 (>= 0.11.0 required), torch 2.10.0+cu128,
            cuda available True, bf16 matmul OK, supports_fp8 False
```
