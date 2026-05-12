# Force-Vector Mask Sweep Results

This note summarizes the force-vector mask-density sweep.

The goal was to test whether physics-aware pretraining helps most when the downstream force-vector task has sparse supervision, or whether it remains useful when many force labels are available.

## Sweep Coverage

The sweep covered:

```text
5 mask levels x 4 pretrained settings = 20 transfer runs
```

Mask levels:

```text
10
25
50
100
all
```

Pretraining settings:

```text
scratch
next_token
next_token_force_law
next_token_force_aux
```

Every expected sweep run has:

```text
metrics_history.json
training_summary.json
best_metrics.json
figs/force_mercury.pdf
figs/force_venus.pdf
figs/force_earth.pdf
figs/force_mars.pdf
figs/force_jupiter.pdf
figs/force_saturn.pdf
figs/force_uranus.pdf
figs/force_neptune.pdf
figs/forces_earth.gif
figs/plot_summary.json
```

No plot artifacts are missing for the 20 expected sweep runs.

The sweep did not include:

```text
next_token_hamiltonian_aux
next_token_force_aux_pretraining
```

So plots for those methods are absent from the mask sweep by design.

## Numerical Results

Best validation force-vector MSE by mask level:

| Mask | Rank 1 | Best Val Loss | Rank 2 | Best Val Loss | Rank 3 | Best Val Loss | Rank 4 | Best Val Loss |
|---:|---|---:|---|---:|---|---:|---|---:|
| 10 | `next_token` | 0.4523 | `next_token_force_law` | 0.5072 | `next_token_force_aux` | 0.5312 | `scratch` | 0.5604 |
| 25 | `next_token_force_law` | 0.4213 | `next_token` | 0.4503 | `next_token_force_aux` | 0.4687 | `scratch` | 0.4730 |
| 50 | `scratch` | 0.2884 | `next_token` | 0.3246 | `next_token_force_law` | 0.4064 | `next_token_force_aux` | 0.4181 |
| 100 | `scratch` | 0.1190 | `next_token` | 0.2411 | `next_token_force_law` | 0.2609 | `next_token_force_aux` | 0.3066 |
| all | `next_token` | 0.000114 | `scratch` | 0.000226 | `next_token_force_aux` | 0.000362 | `next_token_force_law` | 0.000408 |

## Interpretation

The invariant/auxiliary result is positive, but not uniformly dominant.

At sparse label densities:

- `next_token` is best at mask `10`.
- `next_token_force_law` is best at mask `25`.
- All pretrained models beat scratch at mask `10`.
- `next_token_force_law`, `next_token`, and `next_token_force_aux` all beat scratch at mask `25`.

This supports the idea that pretraining helps most when the downstream force-vector task has few labels.

At denser label densities:

- `scratch` wins at mask `50`.
- `scratch` wins at mask `100`.
- All methods reach near-zero error at mask `all`.

This suggests that with enough direct force-vector supervision, scratch training can learn the downstream task without needing physics-aware pretraining.

The cleanest conclusion is:

```text
Physics-aware and next-token pretraining appear to improve sample efficiency in low-label transfer, but the current sweep does not show a robust invariant advantage once direct force-vector labels become dense.
```

## Visual Inspection

The plot artifacts are present and valid.

Qualitative pattern:

- Mask `10`: predictions remain noisy.
- Mask `25`: predictions remain noisy, though the best numerical run is `next_token_force_law`.
- Mask `50`: predictions improve, but are not cleanly radial.
- Mask `100`: predictions improve further, but still show visible angular noise.
- Mask `all`: predicted arrows become cleanly inward/radial for scratch, next-token, and force-law.

The `all` case is mainly a sanity check. It shows that the model and plotting pipeline can represent the correct force field when essentially all force labels are exposed. It should not be interpreted as strong generalization, because the train and validation force-vector target structure is the same solar-system setup.

## What Sparse And Dense Mean Here

Each solar-system two-body sequence contains many possible force-vector labels at planet timesteps. The mask controls how many of those labels are visible during downstream transfer training.

Sparse means:

```text
Only a small number of force vectors are unmasked and contribute to training loss.
```

Dense means:

```text
Many, or all, force vectors are unmasked and contribute to training loss.
```

In this sweep:

```text
mask 10  = 10 unmasked planet-force labels per planet sequence
mask 25  = 25 unmasked planet-force labels per planet sequence
mask 50  = 50 unmasked planet-force labels per planet sequence
mask 100 = 100 unmasked planet-force labels per planet sequence
mask all = every planet-force label is unmasked
```

There are eight solar-system planet sequences. So approximate total supervised force labels are:

```text
mask 10  ~= 80 labels
mask 25  ~= 200 labels
mask 50  ~= 400 labels
mask 100 ~= 800 labels
mask all = all available planet-timestep labels
```

This matters because pretraining is most valuable when the downstream task cannot fully specify the desired function. With only 80 or 200 force labels, the model benefits from prior structure learned during pretraining. With 400, 800, or all labels, the downstream force-vector labels themselves provide enough information for scratch training to catch up or win.

## Current Takeaway

The strongest revised claim is:

```text
The force-law invariant may help in sparse-label force-vector transfer, especially around mask 25, but direct force-vector supervision dominates once enough labels are available.
```

This points toward framing the invariant as a sample-efficiency tool rather than a universal performance booster.

## Recommended Next Step

For a stronger claim, repeat the sparse regimes with multiple random mask seeds:

```text
mask 10, seed 0/1/2/3/4
mask 25, seed 0/1/2/3/4
```

and compare:

```text
scratch
next_token
next_token_force_law
next_token_force_aux
```

If `next_token_force_law` consistently beats scratch and ordinary `next_token` across random sparse masks, then the invariant result becomes much stronger.

## Additional Physics Metrics

MSE is not the only useful comparison. A model can have slightly worse MSE but still produce a more physically structured vector field.

We therefore added a physics-metric evaluator that computes:

```text
direction cosine with the true force vector
radial alignment toward the Sun
central-force torque error
inverse-square consistency
magnitude error
```

These metrics are saved under each run:

```text
outputs/physics/gpt/{run_name}/physics_metrics/summary.json
outputs/physics/gpt/{run_name}/physics_metrics/by_planet.csv
```

They are also folded into:

```text
outputs/physics/gpt/force_mask_sweep/sweep_summary.csv
```

The key interpretive question is whether force-law pretraining has better radial alignment, lower torque, or better inverse-square consistency even when its MSE is not the best. If so, the invariant may be improving physical structure even when the scalar MSE ranking does not show it.
