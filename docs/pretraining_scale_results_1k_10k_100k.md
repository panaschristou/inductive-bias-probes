# Physics-Aware Pretraining Scale and Loss-Weight Study

## Executive Summary

This report summarizes the staged physics-aware pretraining experiments for
force-vector transfer in the orbital-mechanics setting. The experiments compare
ordinary next-token pretraining, Newtonian force-law pretraining, and supervised
force-vector auxiliary pretraining across small and intermediate data regimes.
The primary downstream task is the `mask 25` force-vector transfer task.

The main findings are:

1. Ordinary next-token pretraining improves substantially with scale, moving
   from weak transfer at `1k` to a strong `100k` baseline.
2. The initial `100k` comparison was confounded: ordinary next-token used the
   ordinary `100k` pretraining split, while the physics-aware models used the
   two-body auxiliary dataset because their configs set
   `force_aux_dataset: two_body`.
3. After regenerating the `100k` pretraining split with force/state targets and
   rerunning the physics-aware models with `force_aux_dataset: pretraining`,
   the physics-aware models improved over the matching next-token baseline.
4. The best corrected model is supervised force-vector auxiliary pretraining
   with weight `0.05`. It outperforms the equal-data next-token baseline on
   transfer MSE, component MSE, direction alignment, inward-force fraction, and
   normalized torque error.
5. Loss weight is critical. Force-law weight `0.01` improves over next-token on
   transfer MSE, while force-law weight `0.1` performs poorly. Force-auxiliary
   weight `0.1` is viable but worse than `0.05`.

The corrected interpretation is that the physics signal is useful at `100k`
when trained on the same ordinary pretraining distribution as the baseline.
The earlier conclusion that ordinary next-token dominated at `100k` was due to
a data-source mismatch, not a failure of the physics objectives.

## Experimental Setup

All transfer results reported here use the `mask 25` force-vector transfer
task. In this setting, a limited number of force vectors are revealed in each
solar-system sequence, and the model is evaluated on its ability to recover the
full force-vector field.

The principal methods are:

- Scratch force-vector training
- Ordinary next-token pretraining followed by force-vector transfer
- Next-token pretraining with Newtonian force-law consistency loss followed by
  transfer
- Next-token pretraining with supervised force-vector auxiliary loss followed
  by transfer

The evaluation includes both standard transfer MSE and physical-structure
metrics:

- Component/vector MSE
- Direction cosine with the true force vector
- Radial alignment toward the attracting body
- Inward-force fraction
- Normalized torque error
- Relative magnitude error
- Inverse-square coefficient of variation

Higher is better for direction/radial alignment and inward-force fraction.
Lower is better for MSE, torque error, relative magnitude error, and
inverse-square coefficient of variation.

## Initial Scale Study

The first scale study compared `1k`, `10k`, and `100k` runs. These results are
still useful for understanding the development trajectory, but the `100k`
physics-aware rows must be interpreted with the data-source caveat described
below.

### Transfer MSE

Lower is better.

| Scale | Scratch | Next-token | Force-law | Force-aux |
|---|---:|---:|---:|---:|
| 1k | 0.4730 | 0.4503 | **0.4213** | 0.4687 |
| 10k | 0.4376 | 0.4339 | 0.3469 | **0.3367** |
| 100k | 0.3735 | **0.2014** | 0.2357 | 0.2575 |

Relative to the `1k` result, the initial `100k` transfer MSE improved by:

| Method | MSE Improvement |
|---|---:|
| Scratch | 21.0% |
| Next-token | 55.3% |
| Force-law | 44.1% |
| Force-aux | 45.1% |

### Initial Physics Metrics

| Scale | Method | Direction | Inward Fraction | Torque Error |
|---|---|---:|---:|---:|
| 1k | Scratch | 0.175 | 0.599 | 0.591 |
| 1k | Next-token | 0.201 | 0.616 | 0.589 |
| 1k | Force-law | **0.305** | **0.673** | **0.554** |
| 1k | Force-aux | 0.215 | 0.626 | 0.590 |
| 10k | Scratch | 0.225 | 0.627 | 0.557 |
| 10k | Next-token | 0.238 | 0.639 | 0.581 |
| 10k | Force-law | **0.438** | **0.756** | **0.530** |
| 10k | Force-aux | 0.414 | 0.739 | 0.540 |
| 100k | Scratch | 0.210 | 0.626 | 0.581 |
| 100k | Next-token | **0.687** | **0.912** | **0.484** |
| 100k | Force-law | 0.564 | 0.827 | 0.488 |
| 100k | Force-aux | 0.524 | 0.809 | 0.506 |

The initial scale trend suggested that ordinary next-token pretraining became
dominant at `100k`. However, an audit of the saved configs showed that this
comparison was not equal-data.

## Data-Source Confound

The ordinary next-token `100k` model used:

```text
obs_train.npy
obs_val.npy
```

The initial physics-aware `100k` models used:

```text
obs_two_body_train.npy
obs_two_body_val.npy
full_state_two_body_train.npy
force_vector_two_body_train.npy
```

This happened because the force-aware configs included:

```yaml
force_aux_dataset: two_body
```

As a result, the initial `100k` comparison was closer to:

```text
ordinary 100k next-token pretraining
vs.
two-body force-aware pretraining
```

rather than:

```text
ordinary 100k next-token pretraining
vs.
ordinary 100k next-token pretraining plus physics loss
```

This made the ordinary next-token model appear stronger than the augmented
models because it was the only model receiving the full ordinary `100k`
pretraining distribution.

## Corrected Equal-Data 100k Study

To correct the comparison, the `100k` data was regenerated with
`--save_pretraining_forces`, producing auxiliary targets for the ordinary
pretraining split. The physics-aware models were then trained with:

```text
--force_aux_dataset pretraining
```

The saved configs confirm that the corrected physics-aware runs used:

```text
obs_train.npy
obs_val.npy
full_state_train.npy       for force-law
force_vector_train.npy     for force-auxiliary
```

The corrected comparison therefore uses the same ordinary `100k` pretraining
distribution for the baseline and augmented models.

### Corrected Transfer and Physics Metrics

| Model | Transfer MSE | Component MSE | Direction | Inward Fraction | Torque Error |
|---|---:|---:|---:|---:|---:|
| Scratch | 0.4443 | 0.6259 | 0.213 | 0.617 | 0.572 |
| Next-token | 0.2292 | 0.2610 | 0.599 | 0.866 | 0.514 |
| Force-law, `w=0.01` | 0.1986 | 0.2587 | 0.590 | 0.847 | 0.502 |
| Force-law, `w=0.1` | 0.4632 | 0.5997 | 0.268 | 0.663 | 0.560 |
| Force-aux, `w=0.03` | 0.2330 | 0.2826 | 0.549 | 0.827 | 0.517 |
| Force-aux, `w=0.05` | **0.1845** | **0.2214** | **0.659** | **0.889** | **0.496** |
| Force-aux, `w=0.1` | 0.2585 | 0.2811 | 0.547 | 0.827 | 0.522 |

### Corrected Magnitude and Inverse-Square Metrics

| Model | Relative Magnitude Error | Inverse-Square CV |
|---|---:|---:|
| Scratch | **0.132** | **1.755** |
| Next-token | 0.407 | 1.906 |
| Force-law, `w=0.01` | 0.364 | 1.850 |
| Force-law, `w=0.1` | 0.110 | 1.763 |
| Force-aux, `w=0.03` | 0.389 | 1.877 |
| Force-aux, `w=0.05` | 0.362 | 1.925 |
| Force-aux, `w=0.1` | 0.418 | 1.915 |

The magnitude-style metrics should be interpreted carefully because low
relative magnitude error can occur even when the vector field is poorly
oriented. For example, force-law `w=0.1` has low relative magnitude error but
poor transfer MSE and weak direction alignment. The primary evaluation should
therefore emphasize vector MSE, direction/radial alignment, inward-force
fraction, and torque error.

## Corrected Interpretation

The corrected equal-data runs change the main interpretation.

First, physics-aware pretraining can improve over ordinary next-token
pretraining at the `100k` scale when the augmented models are trained on the
same ordinary pretraining distribution. The best model, force-auxiliary
pretraining with weight `0.05`, improves over the matching next-token baseline
on every primary vector-field metric:

| Metric | Next-token | Force-aux `w=0.05` | Relative Improvement |
|---|---:|---:|---:|
| Transfer MSE | 0.2292 | 0.1845 | 19.5% |
| Component MSE | 0.2610 | 0.2214 | 15.2% |
| Direction alignment | 0.599 | 0.659 | 10.0% |
| Inward-force fraction | 0.866 | 0.889 | 2.7% |
| Torque error | 0.514 | 0.496 | 3.5% |

Second, the force-law objective is highly weight-sensitive. Weight `0.01`
improves transfer MSE relative to next-token, but weight `0.1` performs poorly:

| Model | Transfer MSE | Direction | Inward Fraction |
|---|---:|---:|---:|
| Next-token | 0.2292 | 0.599 | 0.866 |
| Force-law, `w=0.01` | 0.1986 | 0.590 | 0.847 |
| Force-law, `w=0.1` | 0.4632 | 0.268 | 0.663 |

This indicates that a strong fixed force-law penalty can over-regularize or
distort the learned representation. A smaller force-law weight is preferable in
the current setup.

Third, the supervised force-vector auxiliary loss has a clearer useful range.
Weight `0.05` is the strongest overall setting. Weight `0.03` is close to
next-token but does not improve over it, while weight `0.1` is weaker than
`0.05`.

Fourth, the equal-data next-token rerun is weaker than the earlier
`next_token_100k` run (`0.2292` vs. `0.2014` transfer MSE), indicating that
seed/run variability is nontrivial. The corrected result is nevertheless
important because, within the same corrected batch, force-auxiliary `w=0.05`
is the strongest model.

## Conclusions

The corrected `100k` results support the original motivation for adding
physics-aware objectives. The earlier result in which next-token appeared to
dominate was caused by a data-source mismatch. When the augmented models receive
the same ordinary pretraining data and the appropriate auxiliary targets,
physics supervision improves downstream force-vector transfer.

The strongest current setting is:

```text
force_aux_dataset: pretraining
force_loss_weight: 0.05
mask: 25
num_train_trajectories: 100k
```

The best force-law setting tested so far is:

```text
force_aux_dataset: pretraining
force_law_loss_weight: 0.01
mask: 25
num_train_trajectories: 100k
```

The original `0.1` weight is not a good default for force-law in the corrected
equal-data setting.

## Limitations

These results should still be interpreted as an intermediate development study,
not a final benchmark.

- The corrected equal-data comparison appears to use a single seed per
  condition.
- The corrected comparison was evaluated only on `mask 25`.
- The equal-data force-law sweep currently includes `0.01` and `0.1`, but not
  the full set of intermediate values on the ordinary pretraining split.
- The equal-data force-auxiliary sweep includes `0.03`, `0.05`, and `0.1`, but
  should be repeated before selecting a final default.
- The objectives use fixed weights throughout pretraining; scheduled or
  annealed physics losses may perform better.

## Recommended Next Steps

1. **Repeat the corrected `100k` comparison across seeds.** The highest-priority
   runs are ordinary next-token, force-law `0.01`, and force-auxiliary `0.05`.
   This will test whether the corrected ranking is robust.

2. **Evaluate the corrected models across additional masks.** The next masks
   should be `10`, `25`, and `100`. Mask `10` tests sparse-label transfer,
   mask `25` is the current primary regime, and mask `100` provides a
   denser-label control.

3. **Complete the equal-data loss-weight sweep.** For force-law, test
   intermediate weights between `0.01` and `0.1`, especially `0.03` and `0.05`,
   on the ordinary pretraining split. For force-auxiliary, repeat `0.03`,
   `0.05`, and `0.1` with additional seeds.

4. **Test combined low-weight objectives.** Candidate settings are
   `force-law=0.01` with `force-aux=0.03` and `force-law=0.01` with
   `force-aux=0.05`.

5. **Test scheduled physics losses.** A plausible schedule is ordinary
   next-token pretraining for an initial phase followed by a low-weight
   physics-aware phase, or a gradually annealed auxiliary weight.

6. **Scale cautiously.** Before moving directly to `10M` trajectories, run a
   larger intermediate scale such as `500k` or `1M` for the strongest baseline
   and the best corrected physics-aware settings.

## Proposed Shortlist for the Next Experimental Batch

| Model | Rationale |
|---|---|
| `next_token_100k_equal` repeat | Matching ordinary pretraining baseline |
| `next_token_force_aux_100k_equal_w0p05` repeat | Best corrected model |
| `next_token_force_law_100k_equal_w0p01` repeat | Best corrected force-law model |
| Equal-data force-law `w=0.03` | Tests whether an intermediate force-law weight improves over `0.01` |
| Equal-data force-law `w=0.05` | Completes the corrected force-law sweep |
| Combined `force-law=0.01`, `force-aux=0.05` | Tests complementary physics constraints |

This set directly addresses the remaining uncertainty: whether the corrected
physics-aware gain is robust across seeds, masks, and nearby loss weights.
