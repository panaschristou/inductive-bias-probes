# NERSC Full-Scale Physics Scripts

These scripts are for the first full-dataset pass after the 1k pilot sweep.

The code default for `generate_data.py --num_train_trajectories` is
`10_000_000`, so the repository's full pretraining dataset is 10M train
trajectories.

## Recommended Order

1. Estimate storage/runtime:

```bash
python scripts/physics_runtime_estimator.py --observed_hours 30
```

2. Generate the full data on a CPU node:

```bash
sbatch scripts/nersc_01_generate_physics_data.sh
```

For staged scale-up runs, override the scale and trajectory count:

```bash
SCALE_TAG=10k NUM_TRAIN_TRAJECTORIES=10000 sbatch scripts/nersc_01_generate_physics_data.sh
SCALE_TAG=100k NUM_TRAIN_TRAJECTORIES=100000 sbatch scripts/nersc_01_generate_physics_data.sh
```

The default does not pass `--save_pretraining_forces`. That is intentional for
the first full run because ordinary next-token pretraining, force-law
pretraining, two-body force-aux pretraining, and force-vector transfer do not
need full 10M train split force/state/H/L targets. Enable it only if you want
the multi-planet auxiliary-target branch:

```bash
SAVE_PRETRAINING_FORCES=1 sbatch scripts/nersc_01_generate_physics_data.sh
```

3. Pretrain the full models:

```bash
sbatch scripts/nersc_02_pretrain_physics_models.sh
```

For staged scale-up runs:

```bash
SCALE_TAG=10k NUM_DATA_POINTS=10000 MAX_ITERS=1000 EVAL_INTERVAL=50 sbatch scripts/nersc_02_pretrain_physics_models.sh
SCALE_TAG=100k NUM_DATA_POINTS=100000 MAX_ITERS=6250 EVAL_INTERVAL=100 sbatch scripts/nersc_02_pretrain_physics_models.sh
```

This is a Slurm array over:

- `next_token_full`
- `next_token_force_law_full`
- `next_token_force_aux_full`

The default wall time is 48 hours. If a run hits the limit, resubmit with:

```bash
RESUME_FROM_LAST_CKPT=1 sbatch scripts/nersc_02_pretrain_physics_models.sh
```

4. Run the primary mask-25 transfer sweep and physics analysis:

```bash
sbatch scripts/nersc_03_force_mask_sweep.sh
```

For staged scale-up runs:

```bash
SCALE_TAG=10k MASK_VARIANTS=25 MAX_ITERS=5000 sbatch scripts/nersc_03_force_mask_sweep.sh
SCALE_TAG=100k MASK_VARIANTS=25 MAX_ITERS=10000 sbatch scripts/nersc_03_force_mask_sweep.sh
```

For stress/control masks:

```bash
MASK_VARIANTS="10 25 100" sbatch scripts/nersc_03_force_mask_sweep.sh
```

5. Re-run plots/metrics for existing full transfer checkpoints if needed:

```bash
sbatch scripts/nersc_04_plot_metrics.sh
```

## Speed Notes

Data generation already uses Python multiprocessing through
`--num_workers`. The generated `.npy` files are monolithic memmaps, so the
current code is a single-node parallel generator. Using multiple nodes safely
would require a sharded generation mode plus a merge or loader change.

For the current code, the practical speedups are:

- Use a CPU node for generation, not a GPU node.
- Use `NUM_WORKERS` close to available physical cores, for example `120` on a
  128-core node.
- Keep BLAS thread counts at `1` during generation to avoid oversubscription.
- Avoid `--save_pretraining_forces` until we truly need those huge targets.

The first large-scale transfer setting should be `mask 25`; in the pilot sweep
it was the cleanest regime where force-law pretraining improved both MSE and
physics-structure metrics.
