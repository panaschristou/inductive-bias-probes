# Physics Experiment Matrix: 1000-Trajectory Pilot

This matrix is for the pilot experiments generated with:

```text
num_train_trajectories = 1000
```

These are not the final full-dataset runs. The goal is to validate implementation, compare signal direction, and make sure the output artifacts are easy to sync from NERSC before scaling up.

## Output Locations

The code now writes three categories of artifacts.

### Heavy Checkpoints

Training checkpoints are saved under:

```text
inductivebiasprobes/checkpoints/physics/{model_type}/{run_name}/
```

Typical files:

```text
ckpt.pt
last_ckpt.pt
best_val_loss.npy
```

These can be large. Sync them only when we need to resume locally or inspect model weights.

### Lightweight Run Summaries

Each physics training run now also writes compact summaries under:

```text
inductivebiasprobes/outputs/physics/{model_type}/{run_name}/
```

Typical files:

```text
config.json
metrics_history.json
best_metrics.json
best_val_loss.npy
training_summary.json
```

This is the main directory to pull from NERSC for joint analysis. It should be small enough to sync often.

You can override the output directory per run:

```bash
--output_dir /path/to/custom/output/run_name
```

You can disable summary output:

```bash
--no_output_summary
```

### Extrapolations And Figures

White-noise extrapolations are saved under:

```text
inductivebiasprobes/extrapolations/physics/
```

Force-vector plots now write to both a syncable output directory and a local convenience directory:

```text
inductivebiasprobes/outputs/physics/{model_type}/{run_name}/figs/
inductivebiasprobes/experiments/physics/figs/{run_name}/
```

Typical files:

```text
force_mercury.pdf
force_venus.pdf
force_earth.pdf
force_mars.pdf
force_jupiter.pdf
force_saturn.pdf
force_uranus.pdf
force_neptune.pdf
forces_earth.gif
plot_summary.json
```

Some older scripts also write to:

```text
scratch/
llm-prompts/
llm-predictions/
```

Those are not central to the 1000-trajectory pilot unless explicitly running oracle, symbolic regression, or LLM baselines.

## Data Generation Matrix

On NERSC, set the repo root on `PYTHONPATH` before running scripts from `inductivebiasprobes/experiments/physics`:

```bash
cd /pscratch/sd/b/bleach/inductive-bias-probes
export PYTHONPATH="$PWD:${PYTHONPATH}"
cd inductivebiasprobes/experiments/physics
```

Without this, Python only sees the experiment script directory and may not be able to import the top-level `inductivebiasprobes` package.

If the NERSC environment does not have a working Triton install, add `--no_compile` to training commands. Otherwise PyTorch Inductor can fail before the first optimization step with:

```text
RuntimeError: Cannot find a working triton installation
```

| ID | Scope | Command Type | Required For | Status |
|---|---|---|---|---|
| D0 | Existing pilot data | `generate_data.py --num_train_trajectories 1000` | baseline NTP, force aux, force law, force-vector transfer | run if not already present |
| D1 | Current-code pilot data | same as D0, regenerated after our code changes | Hamiltonian/angular-momentum files for two-body splits | required before H/L run |
| D2 | Multi-planet force pilot data | D0 plus `--save_pretraining_forces` | force-aware pretraining on ordinary `train/val` data | optional pilot extension |
| D-full | Full data | full-scale command later | paper-scale or serious final results | later |

Recommended D1 command:

```bash
cd inductivebiasprobes/experiments/physics
python generate_data.py \
  --num_train_trajectories 1000 \
  --num_workers 8
```

Recommended D2 command:

```bash
cd inductivebiasprobes/experiments/physics
python generate_data.py \
  --num_train_trajectories 1000 \
  --num_workers 8 \
  --save_pretraining_forces
```

## Smoke-Test Matrix

Run these before launching real jobs.

| ID | Experiment | Config | Key Data | Expected Output Dir |
|---|---|---|---|---|
| S0 | baseline NTP smoke | `ntp_config` | `obs_train.npy` | `outputs/physics/gpt/next_token/` |
| S1 | force auxiliary smoke | `ntp_force_aux_config` | `obs_two_body_*`, `force_vector_two_body_*` | `outputs/physics/gpt/next_token_force_aux/` |
| S2 | force-law smoke | `ntp_force_law_config` | `obs_two_body_*`, `full_state_two_body_*` | `outputs/physics/gpt/next_token_force_law/` |
| S3 | Hamiltonian/angular smoke | `ntp_hamiltonian_aux_config` | `hamiltonian_two_body_*`, `angular_momentum_two_body_*` | `outputs/physics/gpt/next_token_hamiltonian_aux/` |

Use:

```bash
--max_iters 10 --eval_interval 5 --eval_iters 1 --batch_size 4 --no_wandb --no_compile
```

## Pretraining Matrix

All runs in this section are 1000-trajectory pilot runs.

| ID | Run Name | Config | Objective | Dataset | Checkpoint Dir | Output Dir |
|---|---|---|---|---|---|---|
| P0 | `next_token` | `ntp_config` | next-token CE | ordinary pretraining `obs_train/val` | `checkpoints/physics/gpt/next_token/` | `outputs/physics/gpt/next_token/` |
| P1 | `next_token_force_aux` | `ntp_force_aux_config` | CE + supervised force-vector MSE | two-body | `checkpoints/physics/gpt/next_token_force_aux/` | `outputs/physics/gpt/next_token_force_aux/` |
| P2 | `next_token_force_law` | `ntp_force_law_config` | CE + Newtonian force-law MSE | two-body | `checkpoints/physics/gpt/next_token_force_law/` | `outputs/physics/gpt/next_token_force_law/` |
| P3 | `next_token_hamiltonian_aux` | `ntp_hamiltonian_aux_config` | CE + Hamiltonian MSE + angular momentum MSE | two-body | `checkpoints/physics/gpt/next_token_hamiltonian_aux/` | `outputs/physics/gpt/next_token_hamiltonian_aux/` |
| P4 | `next_token_force_aux_pretraining` | `ntp_force_aux_config` with `--force_aux_dataset pretraining` | CE + supervised force-vector MSE | ordinary multi-planet pretraining with force files | `checkpoints/physics/gpt/next_token_force_aux_pretraining/` | `outputs/physics/gpt/next_token_force_aux_pretraining/` |

Recommended pilot settings:

```bash
--max_iters 1000 --eval_interval 100 --eval_iters 1 --batch_size 64 --no_wandb --no_compile
```

Primary metrics to compare:

```text
val_loss_mean
val_task_loss_mean
val_force_aux_loss_mean
val_force_law_loss_mean
val_hamiltonian_aux_loss_mean
val_angular_momentum_aux_loss_mean
```

Not every metric exists for every run. The run-summary JSON only includes active loss components.

## Transfer Matrix: Force Vector

These evaluate whether physics-aware pretraining improves downstream sparse force-vector learning.

| ID | Run Name | Transfer Source | Command Setting | Checkpoint Dir | Output Dir |
|---|---|---|---|---|---|
| T0 | `force_vector_scratch` | random init | `--pretrained scratch --experiment_name force_vector_scratch` | `checkpoints/physics/gpt/force_vector_scratch/` | `outputs/physics/gpt/force_vector_scratch/` |
| T1 | `next_token_pt_force_vector_transfer` | P0 | `--pretrained next_token` | `checkpoints/physics/gpt/next_token_pt_force_vector_transfer/` | `outputs/physics/gpt/next_token_pt_force_vector_transfer/` |
| T2 | `next_token_force_aux_pt_force_vector_transfer` | P1 | `--pretrained next_token_force_aux` | `checkpoints/physics/gpt/next_token_force_aux_pt_force_vector_transfer/` | `outputs/physics/gpt/next_token_force_aux_pt_force_vector_transfer/` |
| T3 | `next_token_force_law_pt_force_vector_transfer` | P2 | `--pretrained next_token_force_law` | `checkpoints/physics/gpt/next_token_force_law_pt_force_vector_transfer/` | `outputs/physics/gpt/next_token_force_law_pt_force_vector_transfer/` |
| T4 | `next_token_hamiltonian_aux_pt_force_vector_transfer` | P3 | `--pretrained next_token_hamiltonian_aux` | `checkpoints/physics/gpt/next_token_hamiltonian_aux_pt_force_vector_transfer/` | `outputs/physics/gpt/next_token_hamiltonian_aux_pt_force_vector_transfer/` |
| T5 | `next_token_force_aux_pretraining_pt_force_vector_transfer` | P4 | `--pretrained next_token_force_aux_pretraining` | `checkpoints/physics/gpt/next_token_force_aux_pretraining_pt_force_vector_transfer/` | `outputs/physics/gpt/next_token_force_aux_pretraining_pt_force_vector_transfer/` |

Recommended pilot settings:

```bash
--config force_vector_config --max_iters 1000 --eval_interval 50 --eval_iters 1 --batch_size 8 --no_wandb --no_compile
```

Primary transfer metric:

```text
val_loss_mean
```

This is masked force-vector MSE under the existing transfer setup.

## Optional Transfer Matrix: Force Magnitude

Run this after force-vector results if the pretraining effects look promising.

| ID | Run Name | Transfer Source | Notes |
|---|---|---|---|
| M0 | force magnitude from scratch | random init | baseline |
| M1 | force magnitude from `next_token` | P0 | original pretraining |
| M2 | force magnitude from `next_token_force_aux` | P1 | supervised force-vector pretraining |
| M3 | force magnitude from `next_token_force_law` | P2 | equation-based pretraining |
| M4 | force magnitude from `next_token_hamiltonian_aux` | P3 | scalar invariant-target pretraining |

Use:

```bash
--config force_magnitude_config --max_iters 1000 --eval_interval 50 --eval_iters 1 --batch_size 64 --no_wandb --no_compile
```

## Evaluation And Plotting Matrix

| ID | Evaluation | Current Output | Notes |
|---|---|---|---|
| E0 | Training summaries | `outputs/physics/gpt/{run_name}/training_summary.json` | compact sync target |
| E1 | Metric history | `outputs/physics/gpt/{run_name}/metrics_history.json` | compact sync target |
| E2 | Best validation losses | `outputs/physics/gpt/{run_name}/best_val_loss.npy` | compact sync target |
| E3 | Full checkpoint | `checkpoints/physics/gpt/{run_name}/ckpt.pt` | heavy, sync only when needed |
| E4 | Last checkpoint | `checkpoints/physics/gpt/{run_name}/last_ckpt.pt` | heavy, useful for resume |
| E5 | Force plots | compact copy in `outputs/physics/{model_type}/{run_name}/figs/`; local copy in `experiments/physics/figs/{run_name}/` | `plot_forces.py --experiment_name {run_name}` |
| E6 | NTP evaluation | heavy copy in `checkpoints/physics/{model_type}/{run_name}/evaluation/`; compact copy in `outputs/physics/{model_type}/{run_name}/evaluation/` | `evaluate_next_token_prediction.py --experiment_name {run_name}` |

Force plotting examples:

```bash
python plot_forces.py \
  --model_type gpt \
  --experiment_name force_vector_scratch

python plot_forces.py \
  --model_type gpt \
  --experiment_name next_token_pt_force_vector_transfer

python plot_forces.py \
  --model_type gpt \
  --experiment_name next_token_force_aux_pt_force_vector_transfer

python plot_forces.py \
  --model_type gpt \
  --experiment_name next_token_force_law_pt_force_vector_transfer

python plot_forces.py \
  --model_type gpt \
  --experiment_name next_token_hamiltonian_aux_pt_force_vector_transfer

python plot_forces.py \
  --model_type gpt \
  --experiment_name next_token_force_aux_pretraining_pt_force_vector_transfer
```

For quick metric-only testing of the plotting path, skip the GIF:

```bash
python plot_forces.py \
  --model_type gpt \
  --experiment_name next_token_force_aux_pt_force_vector_transfer \
  --skip_animation
```

NTP evaluation example:

```bash
python evaluate_next_token_prediction.py \
  --model_type gpt \
  --experiment_name next_token_force_aux \
  --predict_type next_token \
  --prefix_points 500 \
  --horizons 1 5 100
```

This writes compact metrics to:

```text
outputs/physics/gpt/next_token_force_aux/evaluation/
```

## NERSC Sync Plan

Pull lightweight summaries frequently:

```bash
rsync -avz nersc:/path/to/repo/inductivebiasprobes/outputs/physics/ \
  ./inductivebiasprobes/outputs/physics/
```

Pull selected figures:

```bash
rsync -avz nersc:/path/to/repo/inductivebiasprobes/outputs/physics/gpt/ \
  ./inductivebiasprobes/outputs/physics/gpt/
```

Pull heavy checkpoints only for selected runs:

```bash
rsync -avz nersc:/path/to/repo/inductivebiasprobes/checkpoints/physics/gpt/next_token_force_aux/ \
  ./inductivebiasprobes/checkpoints/physics/gpt/next_token_force_aux/
```

## Completion Criteria For The 1000-Trajectory Pilot

- [ ] D1 data exists with Hamiltonian and angular-momentum files.
- [ ] S0-S3 smoke tests pass.
- [ ] P0-P3 pretraining runs complete.
- [ ] T0-T4 force-vector transfer runs complete.
- [ ] `outputs/physics/gpt/*/training_summary.json` exists for every completed run.
- [ ] `metrics_history.json` exists for every completed run.
- [ ] `outputs/physics/gpt/*/figs/plot_summary.json` exists for every completed force-vector transfer run.
- [ ] We can rank transfer runs by validation loss.
- [ ] We can visually compare true vs predicted force vectors for every completed transfer run.
- [ ] We decide whether P4 multi-planet force-aware pretraining is worth running before full-scale training.
- [ ] We decide which subset should be promoted to full-dataset runs.

## Full-Dataset Follow-Up

After the 1000-trajectory pilot, repeat only the useful subset at larger scale. The full run should use the same run names with a suffix such as:

```text
next_token_full
next_token_force_aux_full
next_token_force_law_full
```

or a separate output root:

```bash
--output_dir /pscratch/.../outputs/physics/gpt/next_token_force_aux_full
```

This keeps pilot and full results separated.
