# Physics Pilot Results: 1000-Trajectory Runs

This note summarizes the first 1000-trajectory pilot for the force-aware and invariant-aware physics extensions.

The purpose of this pilot was not to establish final performance. It was to verify that the implemented objectives train end-to-end, produce checkpoint/output artifacts, transfer into the existing force-vector task, and give us an early signal about which physics constraints are worth scaling.

## Artifact Coverage

The pulled outputs include all planned 1000-trajectory pilot runs.

Pretraining runs:

```text
next_token
next_token_force_aux
next_token_force_law
next_token_hamiltonian_aux
next_token_force_aux_pretraining
```

Force-vector transfer runs:

```text
force_vector_scratch
next_token_pt_force_vector_transfer
next_token_force_aux_pt_force_vector_transfer
next_token_force_law_pt_force_vector_transfer
next_token_hamiltonian_aux_pt_force_vector_transfer
next_token_force_aux_pretraining_pt_force_vector_transfer
```

Each force-vector transfer run has:

```text
metrics_history.json
training_summary.json
best_metrics.json
best_val_loss.npy
figs/force_<planet>.pdf
figs/forces_earth.gif
figs/plot_summary.json
```

So the pilot is complete enough for a first-pass analysis.

## Transfer Results

The clearest comparison is the downstream force-vector transfer task, because all transfer runs are evaluated on the same force-vector target.

Best validation force-vector MSE:

| Rank | Run | Best Val Loss | Best Iteration |
|---:|---|---:|---:|
| 1 | `next_token_force_law_pt_force_vector_transfer` | 0.4788 | 99 |
| 2 | `next_token_pt_force_vector_transfer` | 0.4930 | 799 |
| 3 | `next_token_force_aux_pt_force_vector_transfer` | 0.5019 | 49 |
| 4 | `next_token_force_aux_pretraining_pt_force_vector_transfer` | 0.5286 | 299 |
| 5 | `next_token_hamiltonian_aux_pt_force_vector_transfer` | 0.5856 | 99 |
| 6 | `force_vector_scratch` | 0.6269 | 749 |

The important signal is that every pretrained model beats scratch on best validation loss. That means pretraining is helping the downstream force-vector task, even in a short and noisy pilot.

The strongest result is the Newtonian force-law pretraining run. It beats scratch by about:

```text
(0.6269 - 0.4788) / 0.6269 ~= 23.6%
```

It also slightly beats ordinary next-token pretraining:

```text
(0.4930 - 0.4788) / 0.4930 ~= 2.9%
```

That second margin is small, so we should treat it as an encouraging signal rather than a final conclusion.

## Pretraining Results

Best validation loss during pretraining:

| Run | Best Val Loss | Best Iteration | Notes |
|---|---:|---:|---|
| `next_token_hamiltonian_aux` | 6.6381 | 499 | best scalar validation loss, includes small H/L auxiliary terms |
| `next_token_force_law` | 6.8180 | 899 | best transfer result |
| `next_token_force_aux` | 6.8927 | 899 | force-vector auxiliary supervision on two-body data |
| `next_token` | 7.5609 | 799 | ordinary baseline pretraining |
| `next_token_force_aux_pretraining` | 7.6092 | 999 | multi-planet force targets; did not beat baseline in this pilot |

These pretraining losses are useful but not fully apples-to-apples. The physics-aware runs include extra loss components and, for some configs, use two-body data rather than the ordinary multi-planet pretraining data. The transfer results are the cleaner comparison.

Still, the direction is interesting: the two-body physics-aware runs improve the next-token validation loss relative to the plain baseline in this pilot.

## Plot Inspection

The plot files are present and non-empty for every force-vector transfer run. I spot-checked representative Earth PDFs.

The qualitative result is mixed:

- The plots render correctly.
- The true force arrows point inward toward the Sun, as expected.
- The predicted arrows remain visually noisy.
- Even the best numerical run does not yet show a clean radial force field.

This is important. The auxiliary and invariant losses appear to improve transfer metrics, but the learned force-vector field is not physically clean yet.

## Are Auxiliary Losses And Invariants Helping?

Short answer: yes, there is early evidence that they help, but the evidence is not yet strong enough to declare the approach solved.

What supports “yes”:

- All pretrained transfer runs beat force-vector training from scratch.
- The best transfer result comes from `next_token_force_law`, the run with an equation-based Newtonian force consistency objective.
- The physics-aware two-body pretraining runs also improved pretraining validation loss relative to the plain `next_token` baseline.
- The force-law result directly matches the hypothesis that making the model care about force structure during pretraining can improve later force-vector prediction.

What limits the conclusion:

- The pilot only ran for 1000 iterations.
- Validation curves are noisy.
- Some transfer runs reach their best validation loss very early, then degrade, which suggests sparse-label overfitting.
- The downstream force-vector transfer data uses only 10 unmasked force labels per planet, around 80 sparse force labels total.
- The qualitative plots still show noisy predicted arrows rather than a clean inward force field.

So the right conclusion is:

```text
Physics-aware objectives are giving a positive transfer signal, especially the Newtonian force-law loss, but the current downstream force-vector setup is too sparse and noisy to fully validate physical structure.
```

## Interpretation By Objective

### Supervised Force Auxiliary Loss

`next_token_force_aux` beats scratch after transfer and is close to ordinary next-token pretraining, but it does not beat the force-law run.

This suggests that direct force-vector supervision helps, but the specific two-body force-vector auxiliary setup may not be enough by itself. It may need more label density, better scaling, or a stronger coupling between trajectory prediction and force prediction.

### Newtonian Force-Law Loss

`next_token_force_law` is the most promising run.

It uses the model's predicted force head and compares it to the equation-implied Newtonian force:

```text
F = -G m1 m2 r / ||r||^3
```

This run gives the best downstream force-vector validation loss. That is exactly the kind of signal we wanted from an invariant/physics-structured objective.

### Hamiltonian And Angular Momentum

`next_token_hamiltonian_aux` gives the best pretraining validation loss, but its downstream force-vector transfer is weaker than the force-law and force-auxiliary runs.

This suggests scalar invariant targets may help the sequence model fit the pretraining objective, but they do not automatically produce better force-vector representations. The scalar H/L heads may be too indirect for the force-vector transfer task.

### Multi-Planet Force-Aware Pretraining

`next_token_force_aux_pretraining` did not beat ordinary next-token pretraining in this pilot, either in pretraining validation loss or transfer.

This does not mean the idea is bad. It may mean the multi-planet force targets are harder, noisier, differently scaled, or need longer training. For now, it is lower priority than the two-body force-law path.

## Recommended Next Experiments

The biggest issue is the force-vector transfer label density. The current generator masks almost all force labels:

```text
unmasked_per_sequence = 10
```

With eight planets, this gives roughly 80 unmasked training labels. The transfer models drive train loss near zero while validation remains noisy, which strongly suggests sparse-label overfitting.

Recommended next sweep:

```text
unmasked_per_sequence = 10, 25, 50, 100, all
```

Run this sweep for:

```text
force_vector_scratch
next_token_pt_force_vector_transfer
next_token_force_law_pt_force_vector_transfer
next_token_force_aux_pt_force_vector_transfer
```

If the force-law run continues to outperform as label density increases, then we have a much stronger case that the physics-aware invariant is helping.

## Current Decision

Promote `next_token_force_law` as the leading physics-aware approach for the next iteration.

Keep `next_token_force_aux` as a secondary comparison.

Treat `next_token_hamiltonian_aux` as interesting but not currently validated for force-vector transfer.

Do not prioritize full-scale multi-planet force-aware pretraining until we understand the sparse transfer bottleneck better.
