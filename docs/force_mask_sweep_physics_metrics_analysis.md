# Force Mask Sweep: Physics Metrics Analysis

This note summarizes what we learned after adding physics-structure metrics to the force-vector mask sweep.

The key question was:

```text
Does physics-aware pretraining only improve raw MSE, or does it also produce force vectors that look more physically correct?
```

The answer is more interesting than the MSE-only view:

```text
Force-law pretraining does not always win raw MSE, but it often produces more physically structured force vectors.
```

## Sweep Setup

The downstream force-vector transfer task uses the solar-system two-body force-vector dataset.

Each run receives the full trajectory positions as input, but only a subset of true force-vector labels is visible during training.

Mask levels:

```text
mask 10  = 10 unmasked force labels per planet sequence
mask 25  = 25 unmasked force labels per planet sequence
mask 50  = 50 unmasked force labels per planet sequence
mask 100 = 100 unmasked force labels per planet sequence
mask all = all planet-timestep force labels visible
```

Approximate total labels across eight planet sequences:

```text
mask 10  ~= 80 labels
mask 25  ~= 200 labels
mask 50  ~= 400 labels
mask 100 ~= 800 labels
```

Compared methods:

```text
scratch
next_token
next_token_force_law
next_token_force_aux
```

All methods use the same downstream transfer loss:

```text
masked force-vector MSE
```

The difference is the initialization:

```text
scratch              = random initialization
next_token           = ordinary next-token pretrained checkpoint
next_token_force_law = next-token + Newtonian force-law pretraining
next_token_force_aux = next-token + supervised force-vector auxiliary pretraining
```

## Metrics Added

Raw MSE is useful, but it does not fully describe whether the predicted field is physically meaningful.

We added the following metrics.

### Direction Cosine

Measures whether the predicted force points in the same direction as the true force:

```text
cos(F_pred, F_true)
```

Higher is better. A value near `1` means the predicted force direction matches the true force direction.

### Radial Alignment

Measures whether the predicted force points inward toward the Sun:

```text
cos(F_pred, -r)
```

Higher is better. For a gravitational central force, the value should be near `1`.

### Inward Fraction

Fraction of timesteps where the predicted force has positive inward radial alignment:

```text
cos(F_pred, -r) > 0
```

Higher is better.

### Normalized Torque Error

A central gravitational force should exert near-zero torque about the Sun:

```text
tau_z = r_x F_y - r_y F_x
```

We normalize by `||r|| ||F||`, so this becomes a scale-free central-force error. Lower is better.

### Magnitude Error

Measures whether the predicted force magnitude matches the true force magnitude:

```text
| ||F_pred|| - ||F_true|| | / ||F_true||
```

Lower is better.

### Inverse-Square Consistency

For a fixed planet-star pair, gravity should roughly satisfy:

```text
||F|| * ||r||^2 ~= constant
```

We compute a coefficient of variation for this quantity. Lower is better.

Important caveat: aggregate inverse-square CV across all planets is not very meaningful because different planets have different force scales. The per-planet version is more meaningful.

## Raw MSE Results

Best validation force-vector MSE by mask level:

| Mask | Best Method | Best Val MSE | Notes |
|---:|---|---:|---|
| 10 | `next_token` | 0.4523 | all pretrained methods beat scratch |
| 25 | `next_token_force_law` | 0.4213 | force-law best overall |
| 50 | `scratch` | 0.2884 | scratch catches up and wins |
| 100 | `scratch` | 0.1190 | direct force labels dominate |
| all | `next_token` | 0.000114 | all methods nearly solve the task |

MSE-only interpretation:

```text
Pretraining helps in sparse-label transfer, but scratch catches up once force labels become denser.
```

That is true, but incomplete.

## Physics-Structure Results

The physics metrics show that force-law pretraining often produces a more physically structured force field even when it does not win raw MSE.

### Mask 10

Raw MSE ranking:

```text
next_token best
force_law second
force_aux third
scratch last
```

But force-law is best on physical direction metrics:

| Metric | Best Method | Force-Law Value | Next-Token Value | Scratch Value |
|---|---|---:|---:|---:|
| direction cosine mean | `next_token_force_law` | 0.224 | 0.113 | 0.088 |
| radial alignment mean | `next_token_force_law` | 0.224 | 0.113 | 0.088 |
| inward fraction | `next_token_force_law` | 0.632 | 0.568 | 0.551 |
| normalized torque error | `next_token_force_law` | 0.582 | 0.618 | 0.607 |

Interpretation:

```text
At mask 10, next-token gets better MSE, but force-law produces vectors that point more physically inward.
```

This is an important result. It means MSE alone underestimates the value of the invariant.

### Mask 25

Mask 25 is the clearest win for force-law.

Force-law is best by:

```text
raw MSE
direction cosine
radial alignment
inward fraction
normalized torque error
```

Representative values:

| Metric | Force-Law | Next-Token | Force-Aux | Scratch |
|---|---:|---:|---:|---:|
| raw best val MSE | 0.421 | 0.450 | 0.469 | 0.473 |
| physics component MSE | 0.480 | 0.542 | 0.505 | 0.652 |
| direction cosine mean | 0.305 | 0.201 | 0.215 | 0.175 |
| radial alignment mean | 0.304 | 0.200 | 0.215 | 0.174 |
| inward fraction | 0.673 | 0.616 | 0.626 | 0.599 |
| normalized torque error | 0.554 | 0.589 | 0.590 | 0.591 |

Interpretation:

```text
At mask 25, the force-law invariant improves both accuracy and physical structure.
```

This is the strongest evidence from the sweep.

### Mask 50

Raw MSE is best for scratch:

```text
scratch:    0.288
next_token: 0.325
force_law:  0.406
force_aux:  0.418
```

Physics metrics are mixed:

- `next_token` has best direction cosine and radial alignment.
- `force_aux` has best inward fraction and magnitude error.
- `scratch` has best torque error and inverse-square CV.
- `force_law` is not best here.

Interpretation:

```text
At mask 50, direct labels are becoming strong enough that the force-law pretraining advantage weakens.
```

### Mask 100

Raw MSE is best for scratch:

```text
scratch:    0.119
next_token: 0.241
force_law:  0.261
force_aux:  0.307
```

But force-law is best on direction/radial metrics:

| Metric | Best Method | Force-Law Value | Scratch Value |
|---|---|---:|---:|
| direction cosine mean | `next_token_force_law` | 0.612 | 0.574 |
| radial alignment mean | `next_token_force_law` | 0.612 | 0.574 |
| inward fraction | `next_token_force_law` | 0.839 | 0.810 |

Scratch has slightly better torque error:

```text
scratch torque error:   0.417
force-law torque error: 0.426
```

Interpretation:

```text
At mask 100, scratch wins raw MSE, but force-law still produces the most inward/radially aligned vectors.
```

This supports the idea that the invariant creates a directional physics bias, not necessarily the lowest Euclidean error.

### Mask All

All methods nearly solve the task:

```text
MSE ~ 1e-4 to 4e-4
direction cosine ~ 0.999+
radial alignment ~ 0.999+
inward fraction = 1.0
```

Interpretation:

```text
When every force label is available, all methods can learn the field. This is a sanity check, not the most informative generalization setting.
```

## Main Findings

### 1. Sparse-Label Pretraining Helps

At mask 10 and mask 25, pretrained models beat scratch by raw MSE.

This confirms that pretraining helps when downstream force-vector labels are scarce.

### 2. Force-Law Helps Physical Direction

Force-law does not always win MSE, but it often wins:

```text
direction cosine
radial alignment
inward fraction
```

This is exactly the kind of behavior we hoped an invariant would encourage.

### 3. Mask 25 Is The Cleanest Invariant Win

Mask 25 is the best setting for showing the force-law effect because it wins both:

```text
accuracy metrics
physics-structure metrics
```

Mask 10 is so sparse that ordinary next-token pretraining wins MSE, although force-law still wins direction/radial structure.

Mask 50 and mask 100 have enough direct labels that scratch becomes highly competitive.

### 4. The Invariant Appears To Improve Sample Efficiency And Structure

The force-law invariant is not a universal MSE booster. Instead, it seems to:

```text
help more when labels are limited
push predicted forces toward physically meaningful directions
improve radial/central-force structure in sparse and semi-sparse regimes
```

That is a good result. It means the invariant is acting like an inductive bias, not just an extra regression target.

## Which Mask Should We Use For First Large-Scale Training?

Use:

```text
mask 25
```

as the primary first large-scale transfer setting.

Why mask 25:

- It is still sparse enough that pretraining matters.
- It is not as extremely underdetermined/noisy as mask 10.
- Force-law wins raw MSE at mask 25.
- Force-law also wins the physics metrics at mask 25.
- It is the cleanest setting for testing whether the invariant helps both prediction and physical structure.

The first large-scale comparison should use:

```text
scratch_mask_25
next_token_pt_force_vector_transfer_mask_25
next_token_force_law_pt_force_vector_transfer_mask_25
next_token_force_aux_pt_force_vector_transfer_mask_25
```

If compute allows, also run:

```text
mask 10
```

as a secondary sparse-label stress test.

Mask 10 is useful because it tests extreme sample efficiency, but it is noisier and gives a split result:

```text
next_token wins MSE
force_law wins physical direction metrics
```

Mask 100 is useful as a dense-label sanity/control setting, but it should not be the primary large-scale setting because scratch already becomes very strong there.

## Recommended Large-Scale Plan

Primary:

```text
train large-scale pretraining checkpoints
transfer with mask 25
compare MSE + physics metrics + plots
```

Secondary, if budget permits:

```text
repeat mask 10 for extreme sparse-label transfer
repeat mask 100 as a dense-label control
```

The core paper/project claim should probably be framed as:

```text
Physics-aware force-law pretraining improves force-vector sample efficiency and physical structure in sparse-label transfer.
```

not:

```text
Physics-aware pretraining always minimizes MSE at every label density.
```

The latter is not supported by the sweep. The former is supported by the combination of MSE and physics-structure metrics.
