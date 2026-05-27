# Physics-Aware Pretraining Scale and Loss-Weight Study

## Executive Summary

This report summarizes a staged set of physics-aware pretraining experiments
for force-vector transfer in the orbital-mechanics setting. The study compares
three pretraining-data scales (`1k`, `10k`, and `100k` trajectories), then
performs a `100k` loss-weight sweep for the two physics-aware objectives:
Newtonian force-law consistency and supervised force-vector auxiliary loss.

The main findings are:

1. Scaling next-token pretraining from `1k` to `100k` trajectories substantially
   improves force-vector transfer and physical structure metrics.
2. Physics-aware objectives are most beneficial at smaller pretraining scales,
   especially at `10k`, where both force-law and force-auxiliary pretraining
   outperform ordinary next-token pretraining.
3. At `100k`, ordinary next-token pretraining is the strongest single method on
   the primary vector-field metrics: validation MSE, component MSE, direction
   alignment, radial alignment, inward-force fraction, and torque error.
4. The `100k` weight sweep shows that lower force-law weights improve the
   force-law model relative to its original setting, with `0.01` performing best
   among force-law variants. However, this does not surpass the ordinary
   next-token baseline on the primary transfer metrics.
5. The force-auxiliary objective has a narrower useful range. Weight `0.05`
   gives the best force-auxiliary transfer MSE and the best torque error among
   all models in the weight sweep, while weight `0.03` gives the best
   force-auxiliary direction/radial structure.

Overall, the experiments support the presence of a useful physics signal, but
also show that the current auxiliary objectives do not yet dominate ordinary
next-token pretraining at the `100k` scale. The next stage should therefore
prioritize controlled repeats, lower-weight physics losses, and combined
objectives before moving to a full `10M` trajectory run.

## Experimental Setup

All reported transfer results use the `mask 25` force-vector transfer task. In
this setting, a limited number of force vectors are revealed in each
solar-system sequence, and the model is evaluated on its ability to recover the
full force-vector field.

The scale comparison evaluates:

- Scratch force-vector training
- Ordinary next-token pretraining followed by force-vector transfer
- Next-token pretraining with force-law consistency loss followed by transfer
- Next-token pretraining with supervised force-vector auxiliary loss followed by
  transfer

The scale names denote the number of trajectories used for next-token
pretraining:

- `1k`: pilot-scale pretraining
- `10k`: staged intermediate pretraining
- `100k`: larger staged pretraining

The weight sweep is run at the `100k` scale for:

- Force-law weights: `0.01`, `0.03`, `0.05`, `0.1`
- Force-auxiliary weights: `0.01`, `0.03`, `0.05`, `0.1`

The evaluation includes both standard transfer MSE and physical-structure
metrics:

- Component/vector MSE
- Direction cosine with the true force vector
- Radial alignment toward the attracting body
- Inward-force fraction
- Normalized torque error
- Relative magnitude error
- Inverse-square coefficient of variation

## Scale Study Results

### Transfer MSE

Lower is better.

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.4730 | 0.4503 | **0.4213** | 0.4687 |
| 10k | 0.4376 | 0.4339 | 0.3469 | **0.3367** |
| 100k | 0.3735 | **0.2014** | 0.2357 | 0.2575 |

Relative to the `1k` result, the `100k` transfer MSE improves by:

| Method | MSE Improvement |
|---|---:|
| Scratch | 21.0% |
| Next-token | 55.3% |
| Force-law | 44.1% |
| Force-aux | 45.1% |

The scale trend is clear: larger pretraining datasets improve downstream
force-vector transfer. This effect is strongest for ordinary next-token
pretraining, which becomes the best-performing method at the `100k` scale.

### Direction and Radial Alignment

Higher is better.

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.175 | 0.201 | **0.305** | 0.215 |
| 10k | 0.225 | 0.238 | **0.438** | 0.414 |
| 100k | 0.210 | **0.687** | 0.564 | 0.524 |

At `1k` and `10k`, the force-law objective produces the strongest directional
structure. At `100k`, ordinary next-token pretraining has the strongest
directional and radial alignment.

### Inward-Force Fraction

Higher is better.

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.599 | 0.616 | **0.673** | 0.626 |
| 10k | 0.627 | 0.639 | **0.756** | 0.739 |
| 100k | 0.626 | **0.912** | 0.827 | 0.809 |

The `100k` next-token model produces the most consistently inward-pointing
force field. This indicates that the model is learning meaningful orbital
structure from trajectory prediction alone at this scale.

### Normalized Torque Error

Lower is better.

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.591 | 0.589 | **0.554** | 0.590 |
| 10k | 0.557 | 0.581 | **0.530** | 0.540 |
| 100k | 0.581 | **0.484** | 0.488 | 0.506 |

Torque error follows the same broad trend. Physics-aware losses are helpful at
smaller scales, while ordinary next-token pretraining becomes strongest at
`100k`.

## 100k Loss-Weight Sweep

The `100k` scale results suggested that the original physics-loss weight
(`0.1`) might be too strong once sufficient next-token pretraining data is
available. A weight sweep was therefore run for both force-law and
force-auxiliary objectives.

### Weight-Sweep Transfer Metrics

| Objective | Weight | Transfer MSE | Component MSE | Direction | Inward Fraction | Torque Error |
|---|---:|---:|---:|---:|---:|---:|
| Next-token baseline | - | **0.2014** | **0.2158** | **0.687** | **0.912** | 0.484 |
| Force-law | 0.01 | 0.2145 | 0.2759 | 0.580 | 0.838 | 0.491 |
| Force-law | 0.03 | 0.2470 | 0.3128 | 0.573 | 0.835 | 0.492 |
| Force-law | 0.05 | 0.2342 | 0.3403 | 0.492 | 0.779 | 0.502 |
| Force-law | 0.1 | 0.2230 | 0.3572 | 0.483 | 0.785 | 0.515 |
| Force-aux | 0.01 | 0.2936 | 0.3450 | 0.484 | 0.781 | 0.518 |
| Force-aux | 0.03 | 0.2364 | 0.2966 | 0.548 | 0.820 | 0.505 |
| Force-aux | 0.05 | 0.2107 | 0.3461 | 0.517 | 0.797 | **0.482** |
| Force-aux | 0.1 | 0.2737 | 0.3615 | 0.484 | 0.787 | 0.524 |
| Scratch baseline | - | 0.3735 | 0.5293 | 0.210 | 0.626 | 0.581 |

### Magnitude and Inverse-Square Metrics

| Objective | Weight | Relative Magnitude Error | Inverse-Square CV |
|---|---:|---:|---:|
| Next-token baseline | - | 0.327 | 1.865 |
| Force-law | 0.01 | 0.242 | 1.807 |
| Force-law | 0.03 | 0.304 | 1.858 |
| Force-law | 0.05 | 0.307 | 1.851 |
| Force-law | 0.1 | 0.320 | 1.944 |
| Force-aux | 0.01 | 0.342 | 1.907 |
| Force-aux | 0.03 | 0.249 | 1.799 |
| Force-aux | 0.05 | 0.297 | 1.850 |
| Force-aux | 0.1 | 0.198 | 1.774 |
| Scratch baseline | - | 0.301 | 1.855 |

The magnitude-style metrics present a slightly different picture from the main
vector-field metrics. The default `100k` force-law and force-auxiliary runs
remain strongest on relative magnitude error and inverse-square consistency,
respectively. This suggests that the physics-aware objectives may be improving
some aspects of force magnitude calibration even when they do not improve
overall vector prediction.

## Interpretation

The scale study and weight sweep support three main conclusions.

First, ordinary next-token pretraining is a strong representation-learning
baseline for this problem. At `100k`, it substantially outperforms scratch and
all physics-aware variants on the principal vector-field metrics. This indicates
that trajectory prediction alone can induce meaningful orbital structure when
enough pretraining data is available.

Second, physics-aware objectives remain useful, but their benefit is
regime-dependent. They are clearly beneficial at `10k`, where force-aware
pretraining outperforms ordinary next-token pretraining on transfer MSE and
physical alignment. At `100k`, the same objectives no longer improve the main
vector-field metrics, suggesting either over-regularization, imperfect loss
balancing, or a mismatch between the auxiliary target and the downstream
transfer objective.

Third, the loss-weight sweep partially supports the hypothesis that smaller
weights are preferable at `100k`. For force-law pretraining, weight `0.01`
is the best swept value on transfer MSE, component MSE, direction alignment,
radial alignment, and inward-force fraction. For force-auxiliary pretraining,
weight `0.05` is best on transfer MSE and torque error, while weight `0.03`
is best on component MSE and directional structure. However, none of the
swept physics-aware runs surpass the ordinary next-token baseline on the
primary transfer metrics.

## Limitations

These results should be interpreted as a controlled development study rather
than a final benchmark.

- The experiments appear to use a single seed per condition, so stochastic
  variation is not yet measured.
- The force-aware objectives were only swept over a small set of weights.
- The weight sweep was evaluated only at `mask 25`; other sparse-label regimes
  may respond differently.
- The force-law and force-auxiliary objectives may require different schedules,
  such as warmup, annealing, or delayed activation, rather than a fixed weight
  throughout pretraining.
- The current objectives are evaluated after transfer, so the pretraining loss
  itself is not sufficient to determine downstream quality.

## Recommended Next Steps

1. **Repeat key `100k` conditions across seeds.** The highest-priority repeat
   set is ordinary next-token, force-law `0.01`, force-auxiliary `0.03`, and
   force-auxiliary `0.05`. This will determine whether the observed ranking is
   robust or seed-sensitive.

2. **Evaluate the best weight-sweep models across additional masks.** The next
   masks should be `10`, `25`, and `100`. Mask `10` tests sparse-label
   extrapolation, mask `25` is the current primary regime, and mask `100`
   provides a denser-label control.

3. **Test scheduled physics losses.** Instead of applying the physics loss at a
   fixed weight for the entire run, test delayed or annealed schedules. A
   plausible schedule is ordinary next-token pretraining for an initial phase,
   followed by a low-weight force-law or force-auxiliary phase.

4. **Test combined objectives at low weight.** A combined force-law +
   force-auxiliary model may capture complementary information if both weights
   are kept small. Candidate settings are `(force-law=0.01, force-aux=0.03)`
   and `(force-law=0.01, force-aux=0.05)`.

5. **Run a larger-scale pretraining experiment only after repeats.** The
   current evidence does not yet justify moving directly to `10M` trajectories
   for every condition. A more efficient next scale would be `500k` or `1M`
   using the strongest baseline and two selected physics-aware variants.

6. **Keep ordinary next-token as the primary baseline.** Since the `100k`
   next-token model is currently strongest on the main vector-field metrics,
   all future physics-aware experiments should be compared against it rather
   than only against scratch.

## Proposed Shortlist for the Next Run

The next experimental batch should be:

| Model | Rationale |
|---|---|
| `next_token_100k` repeat | Current strongest baseline |
| `force_law_100k_w0p01` repeat | Best force-law weight on primary structure metrics |
| `force_aux_100k_w0p03` repeat | Best force-aux weight on component MSE and direction |
| `force_aux_100k_w0p05` repeat | Best force-aux weight on transfer MSE and torque |
| combined `force_law=0.01`, `force_aux=0.03` | Tests complementary physics constraints |
| combined `force_law=0.01`, `force_aux=0.05` | Tests complementary physics constraints with stronger auxiliary supervision |

This set is small enough to be computationally manageable while directly
addressing the main open question: whether physics-aware objectives can improve
over ordinary next-token pretraining once the loss weight and objective
combination are tuned.
