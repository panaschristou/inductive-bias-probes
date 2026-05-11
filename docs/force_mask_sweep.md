# Force-Vector Mask Sweep

This sweep tests whether the physics-aware pretraining advantage remains when the downstream force-vector task has more supervised labels.

The previous pilot used:

```text
unmasked_per_sequence = 10
```

That is only about 80 unmasked force labels total across the eight solar-system planet sequences. The sweep creates mask-specific target files and runs transfer for each mask density without overwriting earlier results.

## What The Script Does

Run:

```bash
python run_force_mask_sweep.py
```

The script will:

1. Load `force_vector_solar_system_two_body.npy`.
2. Create mask-specific files:

```text
force_vector_solar_system_two_body_masked_10.npy
force_vector_solar_system_two_body_masked_25.npy
force_vector_solar_system_two_body_masked_50.npy
force_vector_solar_system_two_body_masked_100.npy
force_vector_solar_system_two_body_masked_all.npy
```

3. Run force-vector transfer for:

```text
scratch
next_token
next_token_force_law
next_token_force_aux
```

4. Plot force vectors for each completed transfer run.
5. Write a manifest to:

```text
inductivebiasprobes/outputs/physics/gpt/force_mask_sweep/sweep_manifest.json
```

6. Write compact comparison files to:

```text
inductivebiasprobes/outputs/physics/gpt/force_mask_sweep/sweep_summary.json
inductivebiasprobes/outputs/physics/gpt/force_mask_sweep/sweep_summary.csv
```

Each transfer run also writes its normal output directory:

```text
inductivebiasprobes/outputs/physics/gpt/{run_name}/
```

## NERSC Command

From the repo root:

```bash
cd /pscratch/sd/b/bleach/inductive-bias-probes
export PYTHONPATH="$PWD:${PYTHONPATH}"
cd inductivebiasprobes/experiments/physics
```

Then run:

```bash
python run_force_mask_sweep.py \
  --mask_variants 10 25 50 100 all \
  --pretrained scratch next_token next_token_force_law next_token_force_aux \
  --max_iters 1000 \
  --eval_interval 50 \
  --eval_iters 1 \
  --batch_size 8 \
  --dtype bfloat16 \
  --no_compile \
  --skip_existing
```

If GIF generation is slow, add:

```bash
--skip_animation
```

If you only want to create the mask files and manifest:

```bash
--skip_training --skip_plots
```

For a dry run that prints commands without executing them:

```bash
--dry_run
```

## Run Names

The script names outputs by mask density.

Scratch examples:

```text
force_vector_scratch_mask_10
force_vector_scratch_mask_25
force_vector_scratch_mask_50
force_vector_scratch_mask_100
force_vector_scratch_mask_all
```

Pretrained examples:

```text
next_token_pt_force_vector_transfer_mask_10
next_token_force_law_pt_force_vector_transfer_mask_10
next_token_force_aux_pt_force_vector_transfer_mask_10
```

## Analysis Question

The sweep should answer:

```text
Does physics-aware pretraining help most when downstream force labels are sparse,
or does it remain useful as we expose more force labels?
```

Expected interpretations:

- If `next_token_force_law` remains best across mask densities, the invariant objective is likely learning useful physical structure.
- If all methods converge as labels become dense, the invariant mainly helps low-label transfer.
- If scratch catches up immediately, the earlier gain was mostly sparse-label regularization.
- If predicted arrows become radially inward as labels increase, the qualitative physics story becomes much stronger.
