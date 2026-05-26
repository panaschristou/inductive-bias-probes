#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -A m4698
#SBATCH -J phys_pretrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -G 1
#SBATCH --array=0-2%1
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
RUN_TAG="${RUN_TAG:-${SCALE_TAG}_pretrain_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/slurm}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_TAG}}"
mkdir -p "${OUTDIR}" outputs/slurm

NUM_DATA_POINTS="${NUM_DATA_POINTS:-100000}"
MAX_ITERS="${MAX_ITERS:-6250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
DTYPE="${DTYPE:-bfloat16}"
USE_WANDB="${USE_WANDB:-0}"
NO_COMPILE="${NO_COMPILE:-1}"
RESUME_FROM_LAST_CKPT="${RESUME_FROM_LAST_CKPT:-0}"
INCLUDE_HAMILTONIAN="${INCLUDE_HAMILTONIAN:-0}"

PRIMARY_RUN_SPECS=(
  "next_token_${SCALE_TAG} ntp_config"
  "next_token_force_law_${SCALE_TAG} ntp_force_law_config"
  "next_token_force_aux_${SCALE_TAG} ntp_force_aux_config"
)
HAMILTONIAN_RUN_SPEC="next_token_hamiltonian_aux_${SCALE_TAG} ntp_hamiltonian_aux_config"

RUN_SPECS=("${PRIMARY_RUN_SPECS[@]}")
if [[ "${INCLUDE_HAMILTONIAN}" == "1" ]]; then
  RUN_SPECS+=("${HAMILTONIAN_RUN_SPEC}")
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#RUN_SPECS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; available specs: ${#RUN_SPECS[@]}"
  exit 2
fi
read -r EXPERIMENT_NAME CONFIG_NAME <<< "${RUN_SPECS[${TASK_ID}]}"

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
LOG_FILE="${OUTDIR}/slurm-${SLURM_JOB_NAME:-phys_pretrain}-${EXPERIMENT_NAME}-${STAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ">>> Physics pretraining"
echo "[project] ${PROJECT_ROOT}"
echo "[outdir] ${OUTDIR}"
echo "[run] ${EXPERIMENT_NAME}"
echo "[config] ${CONFIG_NAME}"

CMD=(
  python inductivebiasprobes/experiments/physics/train_model.py
  --config "${CONFIG_NAME}"
  --experiment_name "${EXPERIMENT_NAME}"
  --num_data_points "${NUM_DATA_POINTS}"
  --max_iters "${MAX_ITERS}"
  --eval_interval "${EVAL_INTERVAL}"
  --eval_iters "${EVAL_ITERS}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --dtype "${DTYPE}"
)

if [[ "${USE_WANDB}" != "1" ]]; then
  CMD+=(--no_wandb)
fi
if [[ "${NO_COMPILE}" == "1" ]]; then
  CMD+=(--no_compile)
fi
if [[ "${RESUME_FROM_LAST_CKPT}" == "1" ]]; then
  CMD+=(--resume_from_last_ckpt)
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
