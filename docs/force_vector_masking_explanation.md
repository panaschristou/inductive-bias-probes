# Force-Vector Masking Explanation

This note explains what the force-vector mask means in the downstream transfer task.

## Core Idea

The model always receives the trajectory positions as input:

```text
planet/sun positions over time
```

But during force-vector transfer, it only receives true force-vector labels at selected timesteps.

The masked target file contains:

```text
true force vector     at unmasked timesteps
mask value, inf       at masked timesteps
```

The loss only applies to unmasked force labels. Masked labels are ignored.

## What Mask 10 Means

For each planet sequence:

```text
input:  full orbit trajectory positions
labels: only 10 true force vectors are visible
loss:   only those 10 force vectors count
```

All other force-vector labels are hidden from the loss during training.

Then, during validation and plotting, we ask the model to output force vectors across the full trajectory:

```text
predict force vector at every timestep
```

and compare against the full true force-vector field.

So `mask 10` asks:

```text
Can the model infer the full vector field from only a tiny number of force labels?
```

## What Mask 100 Means

For each planet sequence:

```text
input:  full orbit trajectory positions
labels: 100 true force vectors are visible
loss:   those 100 force vectors count
```

This is an easier task than `mask 10`, because the model receives many more examples of the target force field.

So `mask 100` asks:

```text
Can the model fill in the rest of the vector field when it already sees many force labels?
```

## What Mask All Means

For each planet sequence:

```text
input:  full orbit trajectory positions
labels: every planet-timestep force vector is visible
loss:   all force labels count
```

This is the easiest setting. The model no longer has to infer much from sparse supervision because it is directly trained on the whole available force field.

The `all` case is useful as a sanity check:

```text
Can the model and plotting pipeline represent the correct inward/radial force field when labels are dense?
```

It should not be interpreted as strong generalization by itself.

## Sparse Versus Dense Label Regimes

Sparse labels mean:

```text
Only a small number of force vectors are visible to the transfer loss.
```

Dense labels mean:

```text
Many, or all, force vectors are visible to the transfer loss.
```

In this sweep:

```text
mask 10  = sparse
mask 25  = sparse-to-moderate
mask 50  = moderate
mask 100 = dense
mask all = fully supervised on the available force labels
```

There are eight solar-system planet sequences. Approximate total visible force labels are:

```text
mask 10  ~= 80 labels
mask 25  ~= 200 labels
mask 50  ~= 400 labels
mask 100 ~= 800 labels
mask all = all available planet-timestep labels
```

## Why This Tests Pretraining

The masking sweep is a sample-efficiency test.

With very few force labels, a scratch model has limited information about the target force field. A pretrained model may do better because it already has a useful prior from trajectory pretraining or physics-aware pretraining.

The hoped-for behavior is:

```text
with few labels, physics-aware pretraining helps the model fill in the missing vector field.
```

The observed behavior from the sweep was:

- At `mask 10`, pretraining helps compared to scratch.
- At `mask 25`, force-law pretraining is the best run.
- At `mask 50` and `mask 100`, scratch catches up and wins.
- At `mask all`, every method can learn the force field nearly exactly.

So the refined interpretation is:

```text
Pretraining and invariants appear to help most when force-vector labels are sparse.
When labels are dense, direct force-vector supervision is strong enough for scratch training to catch up.
```

## Important Distinction

Scratch in the sweep does not mean “scratch with different physics losses.”

It means:

```text
random initialization -> train directly on masked force-vector MSE
```

The other transfer runs use the same downstream force-vector MSE loss, but start from different pretrained checkpoints:

```text
next_token
next_token_force_law
next_token_force_aux
```

So the sweep tests whether different pretraining objectives improve downstream force-vector sample efficiency. It does not test whether adding physics losses during transfer itself would help.
