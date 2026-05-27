# Pretraining Scale Results: 1k, 10k, and 100k

This note compares the staged pretraining runs using the `mask 25` force-vector
transfer task. The goal was to test whether we can see the desired behavior
before committing to full 10M-trajectory pretraining.

## Experimental Setup

The comparison uses the same transfer task across scales:

- Force-vector transfer target: `mask 25`
- Transfer task: predict solar-system two-body force vectors from partially
  revealed force labels
- Methods:
  - `scratch`
  - next-token pretraining
  - next-token pretraining with force-law loss
  - next-token pretraining with force-auxiliary loss

The scale names refer to the number of ordinary next-token pretraining
trajectories used before transfer:

- `1k`: original pilot scale
- `10k`: staged scale-up
- `100k`: staged scale-up

## Transfer MSE

Lower is better.

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.4730 | 0.4503 | **0.4213** | 0.4687 |
| 10k | 0.4376 | 0.4339 | 0.3469 | **0.3367** |
| 100k | 0.3735 | **0.2014** | 0.2357 | 0.2575 |

Relative to the 1k transfer result, the 100k runs improve by:

| Method | MSE Improvement |
|---|---:|
| Scratch | 21.0% |
| Next-token | 55.3% |
| Force-law | 44.1% |
| Force-aux | 45.1% |

The main conclusion is that scaling pretraining helps substantially. The
transfer task gets much easier after 100k trajectories, especially for models
with next-token pretraining.

## Physics Metrics

Higher is better for direction/radial alignment and inward fraction. Lower is
better for normalized torque error.

### Direction / Radial Alignment

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.175 | 0.201 | **0.305** | 0.215 |
| 10k | 0.225 | 0.238 | **0.438** | 0.414 |
| 100k | 0.210 | **0.687** | 0.564 | 0.524 |

### Inward Force Fraction

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.599 | 0.616 | **0.673** | 0.626 |
| 10k | 0.627 | 0.639 | **0.756** | 0.739 |
| 100k | 0.626 | **0.912** | 0.827 | 0.809 |

### Normalized Torque Error

Lower is better.

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.591 | 0.589 | **0.554** | 0.590 |
| 10k | 0.557 | 0.581 | **0.530** | 0.540 |
| 100k | 0.581 | **0.484** | 0.488 | 0.506 |

## Interpretation

We are getting the progress and behavior we wanted in the broad sense:

1. More pretraining data improves transfer.
2. Pretrained models beat scratch much more clearly at 100k than at 1k.
3. The force-aware losses help strongly in the 1k and 10k regimes.
4. The learned force fields become much more physically structured at 100k.

The nuance is important: the physics losses do not monotonically dominate at
100k. At 10k, the force-aware objectives are clearly best. At 100k, plain
next-token pretraining becomes strongest on the main vector-field metrics:
MSE, direction/radial alignment, inward fraction, and torque error.

This suggests that the physics signal is real, but the current loss weight
(`0.1`) may be too strong once the model has enough data to learn useful
structure from next-token pretraining alone. The force-law and force-aux models
still have useful behavior, especially on some magnitude-style metrics, but the
main next step should be a loss-weight sweep rather than jumping immediately to
10M trajectories.

## Next Experiment

Run a 100k weight sweep for both force-law and force-aux losses:

- `0.01`
- `0.03`
- `0.05`
- `0.1`

The hypothesis is that lower physics-loss weights may preserve the physical
regularization benefit without constraining the representation too much at
larger pretraining scale.
