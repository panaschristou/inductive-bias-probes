#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -A m4698
#SBATCH -J phys_mask_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -G 1
#SBATCH --mail-user=pc33@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
set -euo pipefail

CPUS_PER_GPU="${CPUS_PER_GPU:-32}"
NUM_GPUS="${NUM_GPUS:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sys_dev}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/pscratch/sd/${USER:0:1}/${USER}}"

cd "${PROJECT_ROOT}"

SCALE_TAG="${SCALE_TAG:-100k}"
RUN_TAG="${RUN_TAG:-${SCALE_TAG}_mask_sweep_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/slurm}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_TAG}}"
mkdir -p "${OUTDIR}" outputs/slurm

MASK_VARIANTS="${MASK_VARIANTS:-25}"
PRETRAINED_RUNS="${PRETRAINED_RUNS:-scratch next_token_${SCALE_TAG} next_token_force_law_${SCALE_TAG} next_token_force_aux_${SCALE_TAG}}"
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-_${SCALE_TAG}}"

MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_ITERS="${EVAL_ITERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bfloat16}"
USE_WANDB="${USE_WANDB:-0}"
NO_COMPILE="${NO_COMPILE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_ANIMATION="${SKIP_ANIMATION:-0}"

export OMP_NUM_THREADS="${CPUS_PER_GPU}"
export OPENBLAS_NUM_THREADS="${CPUS_PER_GPU}"
export MKL_NUM_THREADS="${CPUS_PER_GPU}"
export NUMEXPR_NUM_THREADS="${CPUS_PER_GPU}"
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if command -v module >/dev/null 2>&1; then
  module load python >/dev/null 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${OUTDIR}/slurm-${SLURM_JOB_NAME:-phys_mask_sweep}-${STAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

read -r -a MASK_ARRAY <<< "${MASK_VARIANTS}"
read -r -a PRETRAINED_ARRAY <<< "${PRETRAINED_RUNS}"

echo ">>> Physics force-mask sweep"
echo "[project] ${PROJECT_ROOT}"
echo "[outdir] ${OUTDIR}"
echo "[scale] ${SCALE_TAG}"
echo "[masks] ${MASK_VARIANTS}"
echo "[pretrained] ${PRETRAINED_RUNS}"

CMD=(
  python inductivebiasprobes/experiments/physics/run_force_mask_sweep.py
  --mask_variants "${MASK_ARRAY[@]}"
  --pretrained "${PRETRAINED_ARRAY[@]}"
  --experiment_suffix "${EXPERIMENT_SUFFIX}"
  --max_iters "${MAX_ITERS}"
  --eval_interval "${EVAL_INTERVAL}"
  --eval_iters "${EVAL_ITERS}"
  --batch_size "${BATCH_SIZE}"
  --dtype "${DTYPE}"
  --device cuda
)

if [[ "${USE_WANDB}" == "1" ]]; then
  CMD+=(--use_wandb)
fi
if [[ "${NO_COMPILE}" == "1" ]]; then
  CMD+=(--no_compile)
else
  CMD+=(--compile)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip_existing)
fi
if [[ "${SKIP_ANIMATION}" == "1" ]]; then
  CMD+=(--skip_animation)
fi

GPU_BIND="map_gpu:0"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  GPU_BIND="map_gpu:$(seq -s, 0 $((NUM_GPUS - 1)))"
fi

srun -n 1 \
  --cpus-per-task=$((CPUS_PER_GPU * NUM_GPUS)) \
  --gpus-per-task="${NUM_GPUS}" \
  --gpu-bind="${GPU_BIND}" \
  --cpu-bind=cores \
  "${CMD[@]}"

echo "[done] artifacts in ${OUTDIR}"
