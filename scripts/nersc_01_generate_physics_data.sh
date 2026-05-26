#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -A m4698
#SBATCH -J phys_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mail-user=pc33@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
set -euo pipefail

CPUS_PER_NODE="${CPUS_PER_NODE:-128}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sys_dev}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/pscratch/sd/${USER:0:1}/${USER}}"

cd "${PROJECT_ROOT}"

SCALE_TAG="${SCALE_TAG:-100k}"
RUN_TAG="${RUN_TAG:-${SCALE_TAG}_data_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/slurm}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_TAG}}"
mkdir -p "${OUTDIR}" outputs/slurm

NUM_TRAIN_TRAJECTORIES="${NUM_TRAIN_TRAJECTORIES:-100000}"
NUM_POINTS_PER_TRAJECTORY="${NUM_POINTS_PER_TRAJECTORY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-120}"
TOTAL_FORCE_MAGNITUDES="${TOTAL_FORCE_MAGNITUDES:-10000}"
NUM_UNMASKED_FORCE_MAGNITUDES="${NUM_UNMASKED_FORCE_MAGNITUDES:-9000}"
SAVE_PRETRAINING_FORCES="${SAVE_PRETRAINING_FORCES:-0}"

# The generator uses multiprocessing. Keep BLAS from spawning nested threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
LOG_FILE="${OUTDIR}/slurm-${SLURM_JOB_NAME:-phys_data}-${STAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ">>> Physics data generation"
echo "[project] ${PROJECT_ROOT}"
echo "[outdir] ${OUTDIR}"
echo "[scale] ${SCALE_TAG}"
echo "[num_train_trajectories] ${NUM_TRAIN_TRAJECTORIES}"
echo "[num_workers] ${NUM_WORKERS}"

CMD=(
  python generate_data.py
  --num_train_trajectories "${NUM_TRAIN_TRAJECTORIES}"
  --num_points_per_trajectory "${NUM_POINTS_PER_TRAJECTORY}"
  --num_workers "${NUM_WORKERS}"
  --total_force_magnitudes "${TOTAL_FORCE_MAGNITUDES}"
  --num_unmasked_force_magnitudes "${NUM_UNMASKED_FORCE_MAGNITUDES}"
)

if [[ "${SAVE_PRETRAINING_FORCES}" == "1" ]]; then
  CMD+=(--save_pretraining_forces)
fi

cd "${PROJECT_ROOT}/inductivebiasprobes/experiments/physics"

srun -n 1 \
  --cpus-per-task="${CPUS_PER_NODE}" \
  --cpu-bind=cores \
  "${CMD[@]}"

echo "[done] artifacts in ${OUTDIR}"
