#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -A m4698
#SBATCH -J phys_plot_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -G 1
#SBATCH --array=0-3
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
RUN_TAG="${RUN_TAG:-${SCALE_TAG}_plot_metrics_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/slurm}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_TAG}}"
mkdir -p "${OUTDIR}" outputs/slurm

MASK_VARIANT="${MASK_VARIANT:-25}"
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-_${SCALE_TAG}}"
SKIP_ANIMATION="${SKIP_ANIMATION:-0}"

RUN_NAMES=(
  "force_vector_scratch_mask_${MASK_VARIANT}${EXPERIMENT_SUFFIX}"
  "next_token_${SCALE_TAG}_pt_force_vector_transfer_mask_${MASK_VARIANT}${EXPERIMENT_SUFFIX}"
  "next_token_force_law_${SCALE_TAG}_pt_force_vector_transfer_mask_${MASK_VARIANT}${EXPERIMENT_SUFFIX}"
  "next_token_force_aux_${SCALE_TAG}_pt_force_vector_transfer_mask_${MASK_VARIANT}${EXPERIMENT_SUFFIX}"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#RUN_NAMES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; available runs: ${#RUN_NAMES[@]}"
  exit 2
fi
EXPERIMENT_NAME="${RUN_NAMES[${TASK_ID}]}"

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
LOG_FILE="${OUTDIR}/slurm-${SLURM_JOB_NAME:-phys_plot_metrics}-${EXPERIMENT_NAME}-${STAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ">>> Physics plots and metrics"
echo "[project] ${PROJECT_ROOT}"
echo "[outdir] ${OUTDIR}"
echo "[run] ${EXPERIMENT_NAME}"

PLOT_CMD=(
  python inductivebiasprobes/experiments/physics/plot_forces.py
  --model_type gpt
  --experiment_name "${EXPERIMENT_NAME}"
  --device cuda
)
if [[ "${SKIP_ANIMATION}" == "1" ]]; then
  PLOT_CMD+=(--skip_animation)
fi

METRIC_CMD=(
  python inductivebiasprobes/experiments/physics/evaluate_force_physics_metrics.py
  --model_type gpt
  --experiment_name "${EXPERIMENT_NAME}"
  --device cuda
)

GPU_BIND="map_gpu:0"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  GPU_BIND="map_gpu:$(seq -s, 0 $((NUM_GPUS - 1)))"
fi

srun -n 1 \
  --cpus-per-task=$((CPUS_PER_GPU * NUM_GPUS)) \
  --gpus-per-task="${NUM_GPUS}" \
  --gpu-bind="${GPU_BIND}" \
  --cpu-bind=cores \
  "${PLOT_CMD[@]}"

srun -n 1 \
  --cpus-per-task=$((CPUS_PER_GPU * NUM_GPUS)) \
  --gpus-per-task="${NUM_GPUS}" \
  --gpu-bind="${GPU_BIND}" \
  --cpu-bind=cores \
  "${METRIC_CMD[@]}"

echo "[done] artifacts in ${OUTDIR}"
