# Force-Aware Implementation Inventory

This note lists the implementation pieces needed to extend the physics experiments with force-aware pretraining and later physics-law or conservation-style invariant losses.

## 1. Baseline Training Path

Status: mostly present.

Already present:

- Next-token pretraining on `obs_train.npy`.
- Force-vector transfer training with `force_vector_config.yaml`.
- Shared model initialization, checkpoint loading, batching, and training loop in `src/train_utils.py`.
- Physics-specific training entrypoint in `experiments/physics/train_model.py`.

Needed:

- Clearer experiment naming so baseline checkpoints and force-aware checkpoints do not overwrite each other.
- Small-run command documentation for local experiments generated with `--num_train_trajectories 1000`.

## 2. Force Targets For Pretraining

Status: partially present.

Already present:

- The simulator can compute force vectors through `process_multiplanet_fn_of_state` in `experiments/physics/generate_data.py`.
- Force vectors are saved for `two_body_train`, `two_body_val`, and `two_body_test`.
- Force vectors are saved for solar-system two-body transfer data.

Missing:

- Force vectors are not saved for ordinary `train`, `val`, and `test` next-token pretraining splits because those splits use `skip_states=True`.

Recommended first step:

- Prototype force-aware next-token pretraining on existing `two_body_train` data, because force targets already exist there.

Later step:

- Add an option such as `--save_pretraining_forces` to generate force and state targets for normal multi-planet pretraining splits.

## 3. Auxiliary Force Head

Status: mostly missing.

Already present:

- `Model.forward(..., return_reps=True)` can return hidden representations.
- The model already supports continuous vector outputs during transfer tasks.

Needed:

- Add an auxiliary force prediction head for next-token pretraining.

Desired structure:

```text
hidden representations -> normal output_head -> next-token logits
hidden representations -> force_aux_head -> 2D force vector
```

Possible design:

- Add the force auxiliary head inside `Model`, gated by config fields.
- Alternatively, prototype with a physics-specific wrapper, but this may complicate checkpointing.

Recommended config fields:

```yaml
use_force_aux_loss: true
force_aux_dim: 2
```

## 4. Force-Aware Loss

Status: missing.

Needed objective:

```text
total_loss = next_token_loss + force_loss_weight * force_vector_loss
```

The force loss should:

- Compare predicted force vectors to true force vectors.
- Ignore masked force targets.
- Ignore sun timesteps when force labels are masked.
- Handle `inf` masks safely.

Recommended config fields:

```yaml
use_force_aux_loss: true
force_loss_weight: 0.1
force_aux_mask_id: .inf
```

## 5. Batch And Data Loader Changes

Status: partially present.

Already present:

- `get_batch()` returns `X, Y`.
- For paired tasks, `get_batch()` can load separate target files.
- For next-token prediction, `Y` is derived by shifting the observation sequence.

Needed:

- Force-aware next-token batches need:

```text
X
Y_next_token
Y_force
```

Possible approaches:

- Extend `get_batch()` to optionally return auxiliary targets.
- Add a physics-specific batch helper for force-aware pretraining.

Recommended first implementation:

- Add a carefully gated optional auxiliary target path so ordinary Othello/gridworld training stays unchanged.

## 6. Training Loop Support

Status: partially present.

Already present:

- Centralized training loop.
- Validation loss estimation.
- WandB logging.
- Callback hooks.

Needed:

- Training loop support for auxiliary losses.
- Validation reporting for next-token loss, force loss, and total loss.

Suggested logging:

```text
train/ntp_loss
train/force_loss
train/total_loss
val/ntp_loss
val/force_loss
val/total_loss
```

## 7. Checkpoint Naming And Experiment Routing

Status: missing or implicit.

Already present:

- Checkpoints are organized by model type and predict type.

Current pattern:

```text
checkpoints/physics/{model_type}/{predict_type}
```

Needed:

- Force-aware pretraining should save to a separate checkpoint directory.

Possible directory:

```text
checkpoints/physics/gpt/next_token_force_aux
```

Recommended CLI/config addition:

```text
--experiment_name next_token_force_aux
```

## 8. Transfer From Force-Aware Checkpoint

Status: partially present.

Already present:

- Transfer loading supports arbitrary `--pretrained` strings because choices are not enforced.
- A checkpoint path such as `checkpoints/physics/gpt/{pretrained}/ckpt.pt` can already be loaded.

Needed:

- Ensure force-aware pretraining checkpoint directory is compatible with the existing transfer path.
- Run transfer with:

```bash
python train_model.py --config force_vector_config --pretrained next_token_force_aux
```

## 9. Evaluation And Comparison

Status: partially present.

Already present:

- Force plotting script.
- Validation losses saved to checkpoint directories.
- Oracle and symbolic regression scripts.

Needed:

- A simple comparison script or table for:

```text
scratch -> force_vector
next_token -> force_vector
next_token_force_aux -> force_vector
```

Useful metrics:

- Best validation force MSE.
- Final validation force MSE.
- Convergence speed.
- Force-vector plots by planet/timestep.

## 10. Physics-Law Loss

Status: not present.

Goal:

Add an optional Newtonian force-law consistency loss.

Equation:

```text
F = -G * m1 * m2 * r / ||r||^3
```

Needed:

- Load state or full-state targets alongside observations.
- Compute force from relative position and masses.
- Compare predicted force to equation-implied force.

Recommended config fields:

```yaml
use_force_law_loss: true
force_law_loss_weight: 0.1
```

## 11. Hamiltonian Or Conservation Invariant Loss

Status: mostly missing.

Goal:

Add trajectory-level conservation losses.

Candidate quantities:

```text
Hamiltonian / total energy
Angular momentum
```

Needed:

- Load full state during training, including relative position, relative velocity, planet mass, and sun mass.
- Compute per-timestep conserved quantities.
- Penalize variation across a trajectory or compare predicted-implied quantities to true quantities.

Important note:

- This is more delicate than force-vector supervision because the current model predicts next tokens, not full continuous state. It should come after the force auxiliary path works.

## Recommended Build Order

1. Add checkpoint and experiment naming.
2. Add force-aware config fields.
3. Add auxiliary force head.
4. Add batch loading for force targets.
5. Add masked force auxiliary loss.
6. Train `next_token_force_aux` on existing two-body data.
7. Transfer from `next_token_force_aux` to `force_vector`.
8. Compare against `next_token` and `scratch`.
9. Add Newtonian force-law loss.
10. Add Hamiltonian or angular-momentum invariant losses.

The smallest useful milestone is:

```text
next-token pretraining + supervised force-vector auxiliary loss on existing two-body data
```
