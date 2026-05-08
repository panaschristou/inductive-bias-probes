# Force-Aware Implementation Checklist

## Current Implementation Status

Implemented now:

- Optional force-vector auxiliary head and masked force-vector loss.
- Optional Newtonian force-law loss from `full_state_*` targets.
- Optional Hamiltonian and angular-momentum scalar auxiliary heads.
- Optional auxiliary target loading in the shared batch path.
- Physics training routing for force-aware next-token pretraining.
- Separate experiment/checkpoint naming with `experiment_name`.
- Data-generation option `--save_pretraining_forces` for ordinary train/val/test force and state targets.
- Prototype configs for force auxiliary, force-law, and Hamiltonian/angular-momentum auxiliary pretraining.

Still needs runtime validation in the project environment:

- Model-forward smoke tests with `torch` installed.
- YAML/config loading smoke tests with project dependencies installed.
- Tiny training run on generated data.
- Transfer comparison against scratch and normal `next_token`.

## Phase 0: Baseline Confirmation

- [ ] Confirm small physics data exists from `--num_train_trajectories 1000`.
- [ ] Run next-token pretraining baseline on the small dataset.
- [ ] Run force-vector transfer from scratch.
- [ ] Run force-vector transfer from normal `next_token`.
- [ ] Save baseline commands in docs.
- [ ] Record baseline validation losses.
- [ ] Generate baseline force plots.

## Phase 1: Experiment Naming

- [ ] Add an `--experiment_name` CLI option or equivalent config field.
- [ ] Route force-aware pretraining checkpoints to a separate directory.
- [ ] Ensure baseline `next_token` checkpoints are not overwritten.
- [ ] Confirm transfer loading can use `--pretrained next_token_force_aux`.

## Phase 2: Force-Aware Data Path

- [ ] Decide first prototype dataset: existing `two_body_train` or new force-enabled `train`.
- [ ] Wire force-aware NTP training to observation file.
- [ ] Wire force-aware NTP training to force-vector target file.
- [ ] Confirm force target shape aligns with model timestep outputs.
- [ ] Confirm mask handling for sun timesteps and unavailable force labels.
- [ ] Add a small shape-check/debug command.

## Phase 3: Auxiliary Force Head

- [ ] Add config fields for force auxiliary prediction.
- [ ] Add a force auxiliary head that maps hidden representations to 2D force vectors.
- [ ] Ensure normal model behavior is unchanged when force auxiliary loss is disabled.
- [ ] Ensure checkpoints can save and load force auxiliary head weights.
- [ ] Confirm transfer from force-aware checkpoint can reset or ignore task-specific heads as needed.

## Phase 4: Force Auxiliary Loss

- [ ] Implement masked force-vector MSE or Huber loss.
- [ ] Add `force_loss_weight`.
- [ ] Combine next-token loss and force loss into total loss.
- [ ] Log next-token loss separately from force loss.
- [ ] Validate auxiliary loss on train and validation batches.

## Phase 5: Force-Aware Pretraining

- [ ] Run a tiny debug training job for `next_token_force_aux`.
- [ ] Verify loss decreases.
- [ ] Verify force auxiliary loss is finite and masked correctly.
- [ ] Run small full pretraining on available 1000-trajectory setup.
- [ ] Save checkpoint under force-aware experiment name.

## Phase 6: Transfer Comparison

- [ ] Transfer from `scratch` to `force_vector`.
- [ ] Transfer from normal `next_token` to `force_vector`.
- [ ] Transfer from `next_token_force_aux` to `force_vector`.
- [ ] Compare validation MSE.
- [ ] Compare convergence speed.
- [ ] Generate force-vector plots for each condition.
- [ ] Record results in docs.

## Phase 7: Normal Pretraining Force Targets

- [ ] Add a data-generation option to save states and force vectors for ordinary `train`, `val`, and `test` splits.
- [ ] Keep default data generation lightweight unless the option is enabled.
- [ ] Regenerate small data with force targets.
- [ ] Run force-aware NTP on multi-planet pretraining data.
- [ ] Compare against two-body-only force-aware pretraining.

## Phase 8: Newtonian Force-Law Loss

- [ ] Load full state or required state components in the batch.
- [ ] Implement Newtonian force calculation from relative position and masses.
- [ ] Add `use_force_law_loss`.
- [ ] Add `force_law_loss_weight`.
- [ ] Compare supervised force loss vs force-law loss.
- [ ] Run transfer evaluation from force-law-pretrained checkpoint.

## Phase 9: Conservation Invariant Losses

- [ ] Decide first invariant: Hamiltonian/energy or angular momentum.
- [ ] Confirm required state variables are generated and loaded.
- [ ] Implement invariant computation.
- [ ] Implement trajectory-level variance or consistency penalty.
- [ ] Add config flags and loss weights.
- [ ] Run small debug training.
- [ ] Compare against force auxiliary and force-law variants.

## Phase 10: Cleanup And Documentation

- [ ] Add experiment command docs.
- [ ] Add result logging docs.
- [ ] Add small-data reproducibility instructions.
- [ ] Add tests for force target alignment.
- [ ] Add tests for masked loss behavior.
- [ ] Add tests for Newtonian force calculation.
- [ ] Update README with the new project direction.
