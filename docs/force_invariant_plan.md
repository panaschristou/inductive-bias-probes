# Force-Aware Physics Pretraining Plan

## Current Understanding

This project starts from the `inductivebiasprobes` codebase, which trains sequence models on physics trajectories and then probes whether pretrained representations transfer to physically meaningful targets.

The existing physics workflow has two relevant modes:

- **Next-token pretraining**: the model predicts the next discretized `(x, y)` position token from trajectory observations.
- **Force-vector transfer**: after pretraining, the model can be fine-tuned to output continuous force vectors and is penalized with regression loss against true `force_vector` targets.

So, the repository already supports force-vector prediction during transfer learning, but it does not appear to use force-vector supervision as part of the original next-token pretraining objective.

## Goal

Add physics structure to training by penalizing deviations from force vectors, not only next-token trajectory prediction.

The first version should be a force-aware auxiliary objective:

```text
total_loss = next_token_loss + lambda_force * force_vector_loss
```

where:

- `next_token_loss` is the existing cross-entropy loss for next-position token prediction.
- `force_vector_loss` compares predicted force vectors against true force vectors.
- `lambda_force` controls the strength of the force-aware penalty.

This is not a strict invariant in the mathematical sense, but it is a structured physics loss. Later versions can add true conservation-law losses.

## Suggested Experimental Stages

### Stage 1: Force-Vector Auxiliary Pretraining

Add an auxiliary force head during next-token pretraining.

The model should still learn next-token prediction, but it should also produce force vectors from its hidden representations. Training would optimize both objectives.

Important comparisons:

```text
normal NTP pretraining -> force-vector transfer
force-aware NTP pretraining -> force-vector transfer
```

This tests whether force supervision during pretraining improves the representations used in downstream transfer.

### Stage 2: Force-Law Consistency Loss

Add a physics-equation penalty based on Newtonian gravity:

```text
F = -G * m1 * m2 * r / ||r||^3
```

This can compare model-predicted force vectors to force vectors implied by the known relative position and masses. It is more physics-explicit than ordinary supervised force-vector regression.

### Stage 3: Conservation Invariant Losses

Add losses based on conserved quantities across a trajectory, such as:

```text
Hamiltonian / total energy:
H = kinetic_energy + potential_energy

Angular momentum:
L = r x p
```

For two-body orbital systems, these should remain approximately constant across time. These losses would penalize trajectories or internal predictions that violate conservation structure.

This stage likely requires generating or saving extra target quantities, such as velocities, masses, Hamiltonian values, angular momentum, or per-timestep energy.

## Data Implications

The current large pretraining data generation path skips states and force targets for the ordinary `train`, `val`, and `test` trajectory splits to save space.

To use force-aware pretraining, we need one of the following:

1. Generate force vectors and state arrays for pretraining trajectories.
2. Start with the existing two-body data splits, where force-related targets are already generated.
3. Add a smaller debug/prototype data mode that stores observations, states, and force vectors together.

Since local testing currently uses `--num_train_trajectories 1000`, the best first move is probably to prototype on small data and keep the full-scale path unchanged until the objective works.

## Implementation Direction

Keep the base model architecture mostly unchanged at first. Add the physics-aware objective in the training layer rather than hard-coding it deeply into `Model`.

Likely files to touch:

- `inductivebiasprobes/src/model.py`: optionally expose hidden representations or add a force head.
- `inductivebiasprobes/src/train_utils.py`: support auxiliary losses in the training loop.
- `inductivebiasprobes/experiments/physics/train_model.py`: wire physics-specific target files and loss functions.
- `inductivebiasprobes/experiments/physics/generate_data.py`: optionally save force/state targets for pretraining splits.
- `inductivebiasprobes/configs/physics/*.yaml`: add flags such as `use_force_aux_loss`, `force_loss_weight`, and possibly `force_target_file`.

## Initial Recommendation

Start with Stage 1:

```text
next-token loss + supervised force-vector auxiliary loss
```

Then compare against the existing baseline. Once that works, add force-law or Hamiltonian-style invariants as separate ablations so the source of any improvement is clear.
