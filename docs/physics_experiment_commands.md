# Physics Experiment Commands

Run these commands from the physics experiment directory, but first put the repo root on `PYTHONPATH`.

On NERSC, from your current checkout:

```bash
cd /pscratch/sd/b/bleach/inductive-bias-probes
export PYTHONPATH="$PWD:${PYTHONPATH}"
cd inductivebiasprobes/experiments/physics
```

For a different checkout path, replace `/pscratch/sd/b/bleach/inductive-bias-probes` with your repo root.

If you do not set `PYTHONPATH`, commands run from `inductivebiasprobes/experiments/physics` can fail with:

```text
ModuleNotFoundError: No module named 'inductivebiasprobes'
```

NERSC environments may also lack a working Triton install. If you see:

```text
RuntimeError: Cannot find a working triton installation
```

rerun the training command with:

```bash
--no_compile
```

The current training code also auto-disables `torch.compile` when Triton is missing, but use `--no_compile` explicitly until that patch is on the machine where you are launching jobs.

After that setup, run the experiment commands from:

```bash
cd inductivebiasprobes/experiments/physics
```

Use `--no_wandb` until smoke tests pass. The physics training script currently overrides `wandb_project` and `wandb_entity`, so WandB project/entity CLI knobs are not fully respected yet.

## Data Generation

### Existing Small Baseline Data

If you already generated data with 1000 training trajectories and do not need Hamiltonian/angular-momentum or multi-planet force-aware pretraining yet, you can use the existing data for:

- baseline `ntp_config`
- `ntp_force_aux_config`
- `ntp_force_law_config`
- force-vector transfer

The force auxiliary and force-law configs use the existing two-body files:

```text
obs_two_body_train.npy
force_vector_two_body_train.npy
full_state_two_body_train.npy
```

### Regenerate Small Data

```bash
python generate_data.py \
  --num_train_trajectories 1000 \
  --num_workers 8
```

This regenerates:

```text
obs_train.npy
obs_val.npy
obs_test.npy
obs_two_body_train.npy
obs_two_body_val.npy
obs_two_body_test.npy
force_vector_two_body_train.npy
force_magnitude_two_body_train.npy
full_state_two_body_train.npy
```

With the current code, it also generates:

```text
hamiltonian_two_body_train.npy
angular_momentum_two_body_train.npy
```

and validation/test equivalents.

### Regenerate Small Data With Pretraining Force Targets

```bash
python generate_data.py \
  --num_train_trajectories 1000 \
  --num_workers 8 \
  --save_pretraining_forces
```

This is needed for force-aware objectives on ordinary multi-planet `train/val/test` data. It saves extra files such as:

```text
force_vector_train.npy
full_state_train.npy
hamiltonian_train.npy
angular_momentum_train.npy
```

### Debug Data Generation

```bash
python generate_data.py \
  --num_train_trajectories 20 \
  --num_val_trajectories 10 \
  --num_test_trajectories 10 \
  --total_force_magnitudes 20 \
  --num_unmasked_force_magnitudes 10 \
  --num_workers 1 \
  --debug
```

Use this if multiprocessing or data shapes fail.

## Data Generation Knobs

`--num_points_per_trajectory`

Number of sequence points saved per trajectory. Default is `1000`. Configs assume `block_size: 999`, so changing this may require regenerated configs or CLI overrides.

`--num_train_trajectories`

Number of ordinary pretraining trajectories for `obs_train.npy`. The paper-scale default is very large. Use `1000` for local testing.

`--num_val_trajectories`

Number of ordinary validation trajectories. Default is `300`.

`--num_test_trajectories`

Number of ordinary test trajectories. Default is `300`.

`--dts`

List of timestep sizes in years. Defaults are roughly one week and six months:

```text
7 * 0.0027
6 * 30 * 0.0027
```

The generator encodes the selected `dt` as a special token at the first timestep.

`--num_bins`

Number of bins for discretizing continuous `(x, y)` positions. Default is `7000`. The vocabulary size becomes:

```text
num_bins + 1 + len(dts)
```

With defaults, this is `7003`.

`--total_force_magnitudes`

Number of two-body force-magnitude trajectories to generate for force-magnitude experiments. Default is `10000`.

`--num_unmasked_force_magnitudes`

For force-magnitude data, this controls how many sequences are fully unmasked. After that, only a few force labels are unmasked per sequence.

`--seed`

Base random seed for reproducible data generation.

`--num_workers`

Number of multiprocessing workers. Use a smaller number locally if your machine is struggling.

`--two_body_only`

Makes ordinary generated systems use one planet and one star. Without this, ordinary pretraining samples multi-planet synthetic exoplanet systems.

`--save_pretraining_forces`

New extension knob. Saves states, force vectors, force magnitudes, Hamiltonian, and angular momentum for ordinary `train`, `val`, and `test` splits.

`--debug`

Runs without multiprocessing pools. Useful for debugging errors and stack traces.

## Smoke Tests

### Baseline Next-Token Smoke Test

```bash
python train_model.py \
  --config ntp_config \
  --max_iters 10 \
  --eval_interval 5 \
  --eval_iters 1 \
  --batch_size 4 \
  --no_wandb \
  --no_compile
```

### Force Auxiliary Smoke Test

```bash
python train_model.py \
  --config ntp_force_aux_config \
  --max_iters 10 \
  --eval_interval 5 \
  --eval_iters 1 \
  --batch_size 4 \
  --no_wandb \
  --no_compile
```

### Force-Law Smoke Test

```bash
python train_model.py \
  --config ntp_force_law_config \
  --max_iters 10 \
  --eval_interval 5 \
  --eval_iters 1 \
  --batch_size 4 \
  --no_wandb \
  --no_compile
```

### Hamiltonian/Angular-Momentum Smoke Test

This requires regenerated data with the current code.

```bash
python train_model.py \
  --config ntp_hamiltonian_aux_config \
  --max_iters 10 \
  --eval_interval 5 \
  --eval_iters 1 \
  --batch_size 4 \
  --no_wandb \
  --no_compile
```

## Pretraining Experiments

### Baseline Next-Token Pretraining

```bash
python train_model.py \
  --config ntp_config \
  --max_iters 1000 \
  --eval_interval 100 \
  --eval_iters 1 \
  --batch_size 64 \
  --no_wandb \
  --no_compile
```

Saves to:

```text
checkpoints/physics/gpt/next_token
```

### Force-Vector Auxiliary Pretraining

```bash
python train_model.py \
  --config ntp_force_aux_config \
  --max_iters 1000 \
  --eval_interval 100 \
  --eval_iters 1 \
  --batch_size 64 \
  --no_wandb \
  --no_compile
```

Objective:

```text
next-token cross entropy + lambda_force * force-vector MSE
```

Saves to:

```text
checkpoints/physics/gpt/next_token_force_aux
```

Useful overrides:

```bash
--force_loss_weight 0.01
--force_loss_weight 0.1
--force_loss_weight 1.0
```

### Newtonian Force-Law Pretraining

```bash
python train_model.py \
  --config ntp_force_law_config \
  --max_iters 1000 \
  --eval_interval 100 \
  --eval_iters 1 \
  --batch_size 64 \
  --no_wandb \
  --no_compile
```

Objective:

```text
next-token cross entropy + lambda_law * MSE(F_pred, -G*m1*m2*r/||r||^3)
```

Saves to:

```text
checkpoints/physics/gpt/next_token_force_law
```

Useful overrides:

```bash
--force_law_loss_weight 0.01
--force_law_loss_weight 0.1
--force_law_loss_weight 1.0
--no-normalize_force_law
```

### Hamiltonian And Angular-Momentum Auxiliary Pretraining

Requires regenerated data with the current generator.

```bash
python train_model.py \
  --config ntp_hamiltonian_aux_config \
  --max_iters 1000 \
  --eval_interval 100 \
  --eval_iters 1 \
  --batch_size 64 \
  --no_wandb \
  --no_compile
```

Objective:

```text
next-token cross entropy
+ lambda_H * MSE(H_pred, H_true)
+ lambda_L * MSE(L_pred, L_true)
```

Saves to:

```text
checkpoints/physics/gpt/next_token_hamiltonian_aux
```

Useful overrides:

```bash
--hamiltonian_loss_weight 0.01
--hamiltonian_loss_weight 0.1
--angular_momentum_loss_weight 0.01
--angular_momentum_loss_weight 0.1
```

### Multi-Planet Force-Aware Pretraining

Requires:

```bash
python generate_data.py \
  --num_train_trajectories 1000 \
  --num_workers 8 \
  --save_pretraining_forces
```

Then run:

```bash
python train_model.py \
  --config ntp_force_aux_config \
  --force_aux_dataset pretraining \
  --experiment_name next_token_force_aux_pretraining \
  --max_iters 1000 \
  --eval_interval 100 \
  --eval_iters 1 \
  --batch_size 64 \
  --no_wandb \
  --no_compile
```

Saves to:

```text
checkpoints/physics/gpt/next_token_force_aux_pretraining
```

## Transfer Experiments

These compare downstream force-vector learning from different pretrained checkpoints.

### Scratch To Force Vector

```bash
python train_model.py \
  --config force_vector_config \
  --pretrained scratch \
  --experiment_name force_vector_scratch \
  --max_iters 1000 \
  --eval_interval 50 \
  --eval_iters 1 \
  --batch_size 8 \
  --no_wandb \
  --no_compile
```

### Normal Next-Token To Force Vector

```bash
python train_model.py \
  --config force_vector_config \
  --pretrained next_token \
  --max_iters 1000 \
  --eval_interval 50 \
  --eval_iters 1 \
  --batch_size 8 \
  --no_wandb \
  --no_compile
```

### Force-Auxiliary Pretraining To Force Vector

```bash
python train_model.py \
  --config force_vector_config \
  --pretrained next_token_force_aux \
  --max_iters 1000 \
  --eval_interval 50 \
  --eval_iters 1 \
  --batch_size 8 \
  --no_wandb \
  --no_compile
```

### Force-Law Pretraining To Force Vector

```bash
python train_model.py \
  --config force_vector_config \
  --pretrained next_token_force_law \
  --max_iters 1000 \
  --eval_interval 50 \
  --eval_iters 1 \
  --batch_size 8 \
  --no_wandb \
  --no_compile
```

### Hamiltonian/Angular-Momentum Pretraining To Force Vector

```bash
python train_model.py \
  --config force_vector_config \
  --pretrained next_token_hamiltonian_aux \
  --max_iters 1000 \
  --eval_interval 50 \
  --eval_iters 1 \
  --batch_size 8 \
  --no_wandb \
  --no_compile
```

### Multi-Planet Force-Auxiliary Pretraining To Force Vector

```bash
python train_model.py \
  --config force_vector_config \
  --pretrained next_token_force_aux_pretraining \
  --max_iters 1000 \
  --eval_interval 50 \
  --eval_iters 1 \
  --batch_size 8 \
  --no_wandb \
  --no_compile
```

## Optional Force-Magnitude Transfer

The same pretrained checkpoints can be tested on force magnitude:

```bash
python train_model.py \
  --config force_magnitude_config \
  --pretrained next_token_force_aux \
  --max_iters 1000 \
  --eval_interval 50 \
  --eval_iters 1 \
  --batch_size 64 \
  --no_wandb \
  --no_compile
```

## Force-Vector Plots

After each force-vector transfer run completes, generate the essential true-vs-predicted force-vector visualizations:

```bash
python plot_forces.py \
  --model_type gpt \
  --experiment_name next_token_force_aux_pt_force_vector_transfer
```

This writes PDFs for all eight planets plus an Earth GIF by default. Outputs are duplicated into:

```text
inductivebiasprobes/outputs/physics/gpt/{run_name}/figs/
inductivebiasprobes/experiments/physics/figs/{run_name}/
```

The `outputs/` copy is the one to sync from NERSC for local analysis. The `figs/` copy is for quick inspection while working inside `inductivebiasprobes/experiments/physics`.

Run the plotting command for each completed force-vector transfer:

```bash
python plot_forces.py --model_type gpt --experiment_name force_vector_scratch
python plot_forces.py --model_type gpt --experiment_name next_token_pt_force_vector_transfer
python plot_forces.py --model_type gpt --experiment_name next_token_force_aux_pt_force_vector_transfer
python plot_forces.py --model_type gpt --experiment_name next_token_force_law_pt_force_vector_transfer
python plot_forces.py --model_type gpt --experiment_name next_token_hamiltonian_aux_pt_force_vector_transfer
python plot_forces.py --model_type gpt --experiment_name next_token_force_aux_pretraining_pt_force_vector_transfer
```

Useful plotting knobs:

`--skip_animation`

Skip the GIF and only write the eight PDF figures. This is useful for fast checks or headless job environments where GIF generation is slow.

`--animation_planet_idx`

Planet index used for the GIF. The default is `2`, which corresponds to Earth.

`--checkpoint_path`

Direct path to a specific checkpoint file, such as `ckpt.pt` or `last_ckpt.pt`. Use this when plotting a checkpoint that does not live at the default `checkpoints/physics/{model_type}/{run_name}/ckpt.pt` location.

`--output_dir`

Override the syncable output directory. By default:

```text
inductivebiasprobes/outputs/physics/{model_type}/{run_name}/figs/
```

`--fig_dir`

Override the local convenience figure directory. By default:

```text
inductivebiasprobes/experiments/physics/figs/{run_name}/
```

## Training Knobs

`--config`

Name of the YAML file in `inductivebiasprobes/configs/physics`, without `.yaml`.

`--model_type`

One of:

```text
gpt, mamba, mamba2, rnn, lstm
```

Default is `gpt`.

`--pretrained`

Checkpoint directory name to load from:

```text
checkpoints/physics/{model_type}/{pretrained}/ckpt.pt
```

Use `scratch` for random initialization.

`--experiment_name`

Checkpoint directory name for the current run. This avoids overwriting baseline checkpoints.

`--force_aux_dataset`

Controls which data split supplies auxiliary targets:

```text
two_body
pretraining
```

`two_body` uses `obs_two_body_*`, `force_vector_two_body_*`, and `full_state_two_body_*`.

`pretraining` uses ordinary `obs_train/val` plus force/state files generated with `--save_pretraining_forces`.

`--max_iters`

Number of training iterations.

`--eval_interval`

Evaluate every N iterations.

`--eval_iters`

Number of batches used during evaluation.

`--batch_size`

Batch size per optimizer step before gradient accumulation.

`--gradient_accumulation_steps`

Number of micro-steps accumulated before optimizer update.

`--learning_rate`

AdamW learning rate.

`--decay_lr`

Enable cosine learning-rate schedule with warmup.

`--device`

Default is `cuda`. Use `--device cpu` only for tiny tests.

`--dtype`

One of:

```text
float32, bfloat16, float16
```

`--no_compile`

Disables `torch.compile`. Auxiliary-loss runs already disable compilation internally, but this is useful for smoke tests.

`--no_wandb`

Disables WandB logging. Recommended for smoke tests.

## Suggested Run Order

1. Generate or confirm small data.
2. Run all smoke tests with `--no_wandb --no_compile`.
3. Train baseline `ntp_config`.
4. Train `ntp_force_aux_config`.
5. Train `ntp_force_law_config`.
6. Regenerate data if needed and train `ntp_hamiltonian_aux_config`.
7. Transfer each checkpoint to `force_vector`.
8. Compare validation losses and force plots.

Core comparison:

```text
scratch -> force_vector
next_token -> force_vector
next_token_force_aux -> force_vector
next_token_force_law -> force_vector
next_token_hamiltonian_aux -> force_vector
```
