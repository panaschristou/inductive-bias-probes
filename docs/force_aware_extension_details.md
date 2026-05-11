# Force-Aware Physics Extension Details

## Purpose

This project extends the original physics experiments by adding physics-aware auxiliary objectives during next-token pretraining. The original code trains a sequence model to predict future trajectory tokens, then fine-tunes that pretrained model on downstream physics targets such as force vectors. Our extension lets the model see physics structure during pretraining itself.

The first added objective is supervised force-vector prediction. We also added hooks for a Newtonian force-law loss and scalar Hamiltonian/angular-momentum auxiliary losses.

## What Existed Before

The original physics workflow had three main data and training regimes.

### 1. Ordinary Next-Token Pretraining

The main pretraining files are:

```text
inductivebiasprobes/data/physics/obs_train.npy
inductivebiasprobes/data/physics/obs_val.npy
inductivebiasprobes/data/physics/obs_test.npy
```

These contain tokenized trajectories. Each timestep has two channels, corresponding to discretized `(x, y)` positions. The model receives a prefix:

```text
X = obs[:, :block_size]
```

and predicts:

```text
Y = obs[:, 1:block_size + 1]
```

The training loss is next-token cross entropy.

For the ordinary `train`, `val`, and `test` splits, the original generator skipped state and force arrays:

```python
skip_states = 'two_body' not in label
```

So ordinary pretraining learned trajectory prediction without direct access to true forces, velocities, masses, Hamiltonians, or angular momentum.

### 2. Two-Body Force Data

The generator also creates special two-body datasets:

```text
obs_two_body_train.npy
obs_two_body_val.npy
obs_two_body_test.npy
state_two_body_train.npy
full_state_two_body_train.npy
force_vector_two_body_train.npy
force_magnitude_two_body_train.npy
```

These are not arbitrary planet-planet pairs. They are classical two-body systems: one lighter body and one heavier star/sun. The two-body data already included true force targets before our extension.

The state arrays store physical variables:

```text
state_*:
  relative x/y position
  relative x/y velocity
  relative mass

full_state_*:
  relative x/y position
  relative x/y velocity
  light-body mass
  heavy-body mass
```

### 3. Solar-System Transfer Data

The force-vector transfer experiment uses:

```text
obs_solar_system_two_body.npy
state_solar_system_two_body.npy
full_state_solar_system_two_body.npy
force_vector_solar_system_two_body.npy
force_vector_solar_system_two_body_masked.npy
```

This is built as eight separate planet-sun two-body systems: Mercury/Sun, Venus/Sun, Earth/Sun, and so on. Most force-vector labels are masked during transfer training. This tests whether a pretrained model can infer/extrapolate force vectors from sparse labels.

## What We Added

### 1. Auxiliary Force Head

The shared `Model` class now supports an optional force head:

```text
hidden representations -> force_aux_head -> 2D force vector
```

The ordinary output head remains unchanged:

```text
hidden representations -> output_head -> next-token logits
```

This means force-aware pretraining can optimize both next-token prediction and force-vector prediction without replacing the original next-token objective.

The total supervised force auxiliary objective is:

```text
L_total = L_next_token + lambda_force * L_force_aux
```

where:

```text
L_force_aux = masked_mse(F_pred, F_true)
```

Force targets can contain `inf` masks. The loss ignores masked entries, including sun/star timesteps and unavailable labels.

### 2. Newtonian Force-Law Loss

We added an optional equation-based force-law loss using `full_state_*` targets.

For relative position vector `r`, light mass `m1`, heavy mass `m2`, and gravitational constant `G`, the Newtonian gravitational force on the light body is:

```text
F(r) = -G * m1 * m2 * r / ||r||^3
```

The implemented loss compares the model's auxiliary force prediction to the force implied by this equation:

```text
L_force_law = masked_mse(F_pred, F_law(r, m1, m2))
```

The force-law target is optionally normalized per sequence:

```text
F_law_normalized = F_law / max_t |F_law_t|
```

This matches the existing force-vector convention, where generated force vectors are normalized by maximum absolute component. The config knob is:

```yaml
normalize_force_law: true
```

### 3. Hamiltonian Auxiliary Target

We added generation and loading support for a scalar Hamiltonian/energy-like target derived from full state.

Using reduced mass:

```text
mu = (m1 * m2) / (m1 + m2)
```

relative velocity `v`, relative radius `||r||`, and gravitational constant `G`, the generated Hamiltonian target is:

```text
H = 0.5 * mu * ||v||^2 - G * m1 * m2 / ||r||
```

The model can train a scalar auxiliary head:

```text
hidden representations -> hamiltonian_aux_head -> H_pred
```

with:

```text
L_H = masked_mse(H_pred, H_true)
```

This is not yet a conservation-variance penalty. It is a supervised scalar auxiliary target. A future conservation loss could penalize variation of `H_t` across a trajectory:

```text
L_H_conservation = variance_t(H_t)
```

### 4. Angular-Momentum Auxiliary Target

We also added a scalar angular-momentum target. In the 2D orbital plane, the z-component of angular momentum is:

```text
L_z = mu * (r_x * v_y - r_y * v_x)
```

The model can train:

```text
hidden representations -> angular_momentum_aux_head -> L_z_pred
```

with:

```text
L_L = masked_mse(L_z_pred, L_z_true)
```

Again, this is currently supervised auxiliary prediction. A later invariant-style conservation loss could penalize variation:

```text
L_L_conservation = variance_t(L_z_t)
```

## Data Generation Changes

Before the extension, ordinary pretraining splits saved only observations and body-count metadata. Two-body splits saved states and forces.

We added:

```bash
--save_pretraining_forces
```

to `experiments/physics/generate_data.py`.

When this flag is enabled, ordinary `train`, `val`, and `test` splits also save:

```text
state_train.npy
full_state_train.npy
force_vector_train.npy
force_magnitude_train.npy
hamiltonian_train.npy
angular_momentum_train.npy
```

and corresponding validation/test files.

The existing two-body splits now also save:

```text
hamiltonian_two_body_train.npy
angular_momentum_two_body_train.npy
```

plus validation/test equivalents.

This matters because:

- `ntp_force_aux_config.yaml` can use existing `force_vector_two_body_*` files.
- `ntp_force_law_config.yaml` can use existing `full_state_two_body_*` files.
- `ntp_hamiltonian_aux_config.yaml` requires the newly generated `hamiltonian_two_body_*` and `angular_momentum_two_body_*` files, so data must be regenerated with the current code.
- Multi-planet force-aware pretraining requires regenerating data with `--save_pretraining_forces`.

## Training Changes

The existing physics training entrypoint remains:

```text
inductivebiasprobes/experiments/physics/train_model.py
```

We added routing for auxiliary objectives. If the config enables one of:

```yaml
use_force_aux_loss: true
use_force_law_loss: true
use_hamiltonian_aux_loss: true
use_angular_momentum_aux_loss: true
```

then `train_model.py` attaches the relevant target files before calling the existing shared `train(...)` loop.

The shared batch path now returns either:

```python
X, Y
```

for ordinary training, or:

```python
X, Y, aux
```

for auxiliary-loss training.

The ordinary baseline path is preserved. For example:

```bash
python train_model.py --config ntp_config
```

still trains ordinary next-token prediction.

## Prototype Configs Added

### `ntp_force_aux_config.yaml`

Uses:

```text
next-token loss + supervised force-vector auxiliary loss
```

Default data:

```text
force_aux_dataset: two_body
```

This uses existing two-body force-vector targets.

### `ntp_force_law_config.yaml`

Uses:

```text
next-token loss + Newtonian force-law loss
```

Default data:

```text
full_state_two_body_train.npy
full_state_two_body_val.npy
```

### `ntp_hamiltonian_aux_config.yaml`

Uses:

```text
next-token loss
+ supervised Hamiltonian auxiliary loss
+ supervised angular-momentum auxiliary loss
```

This requires regenerated data with the current code so the Hamiltonian and angular-momentum files exist.

## How This Differs From The Original Code

Original:

```text
pretrain: next-token only
transfer: force-vector or force-magnitude regression
```

Extended:

```text
pretrain: next-token + optional physics auxiliary objectives
transfer: unchanged, but can start from physics-aware checkpoints
```

The key experimental question is whether physics-aware pretraining improves downstream force-vector transfer.

The core comparison is:

```text
scratch -> force_vector
next_token -> force_vector
next_token_force_aux -> force_vector
next_token_force_law -> force_vector
next_token_hamiltonian_aux -> force_vector
```

## Important Notes

- The current Hamiltonian and angular-momentum additions are supervised auxiliary targets, not strict conservation losses.
- The force-law loss computes a Newtonian target from `full_state_*`.
- Force-vector targets are normalized by max absolute force component in the data generator.
- The physics training script currently hardcodes WandB project/entity after config loading. Use `--no_wandb` for clean local testing unless that is patched.
