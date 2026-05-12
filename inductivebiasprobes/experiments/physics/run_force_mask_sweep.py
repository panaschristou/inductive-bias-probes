import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[2]
PHYSICS_DATA_DIR = BASE_DIR / "data" / "physics"
PHYSICS_OUTPUT_DIR = BASE_DIR / "outputs" / "physics"


DEFAULT_MASK_VARIANTS = ["10", "25", "50", "100", "all"]
DEFAULT_PRETRAINED_RUNS = [
    "scratch",
    "next_token",
    "next_token_force_law",
    "next_token_force_aux",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create force-vector mask variants, run transfer sweeps, plot force "
            "predictions, and save a manifest under outputs/physics."
        )
    )
    parser.add_argument(
        "--mask_variants",
        nargs="+",
        default=DEFAULT_MASK_VARIANTS,
        help='Mask densities to run, e.g. "10 25 50 100 all".',
    )
    parser.add_argument(
        "--pretrained",
        nargs="+",
        default=DEFAULT_PRETRAINED_RUNS,
        help=(
            "Pretrained checkpoint names to transfer from. Use scratch for the "
            "random-init baseline."
        ),
    )
    parser.add_argument("--model_type", type=str, default="gpt")
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Manifest/output directory. Defaults to "
            "outputs/physics/{model_type}/force_mask_sweep."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a transfer run when its training_summary.json already exists.",
    )
    parser.add_argument(
        "--force_regenerate_masks",
        action="store_true",
        help="Regenerate mask files even if they already exist.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Only generate mask files and manifest; do not launch training.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Do not run plot_forces.py after each completed transfer.",
    )
    parser.add_argument(
        "--skip_physics_metrics",
        action="store_true",
        help="Do not run evaluate_force_physics_metrics.py after each completed transfer.",
    )
    parser.add_argument(
        "--skip_animation",
        action="store_true",
        help="Pass --skip_animation to plot_forces.py.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable WandB. By default the sweep passes --no_wandb.",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Enable torch.compile. By default the sweep passes --no_compile.",
    )
    parser.add_argument(
        "--no_compile",
        dest="compile",
        action="store_false",
        help="Keep torch.compile disabled. This is the default.",
    )
    parser.set_defaults(compile=False)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and write manifest without executing training/plotting.",
    )
    return parser.parse_args()


def parse_mask_variant(variant, num_planet_timesteps):
    if str(variant).lower() == "all":
        return num_planet_timesteps
    value = int(variant)
    if value <= 0:
        raise ValueError(f"Mask variant must be positive or 'all', got {variant!r}")
    return min(value, num_planet_timesteps)


def mask_suffix(variant):
    return str(variant).lower()


def masked_force_path(variant):
    return (
        PHYSICS_DATA_DIR
        / f"force_vector_solar_system_two_body_masked_{mask_suffix(variant)}.npy"
    )


def make_mask_variant(variant, seed=0, force=False):
    import numpy as np

    full_path = PHYSICS_DATA_DIR / "force_vector_solar_system_two_body.npy"
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full force-vector target file: {full_path}")

    out_path = masked_force_path(variant)
    if out_path.exists() and not force:
        logger.info("Mask variant %s already exists at %s", variant, out_path)
        return out_path

    force_vectors = np.load(full_path)
    masked = np.full_like(force_vectors, np.inf)
    odd_indices = np.arange(1, force_vectors.shape[1], 2)
    unmasked_per_sequence = parse_mask_variant(variant, len(odd_indices))
    rng = np.random.RandomState(seed)

    for seq_idx in range(force_vectors.shape[0]):
        if unmasked_per_sequence == len(odd_indices):
            unmasked_indices = odd_indices
        else:
            unmasked_indices = rng.choice(
                odd_indices,
                size=unmasked_per_sequence,
                replace=False,
            )
        masked[seq_idx, unmasked_indices, :] = force_vectors[
            seq_idx, unmasked_indices, :
        ]

    np.save(out_path, masked.astype(np.float32, copy=False))
    logger.info(
        "Saved mask variant %s with %s unmasked planet labels per sequence to %s",
        variant,
        unmasked_per_sequence,
        out_path,
    )
    return out_path


def experiment_name(pretrained, variant):
    suffix = mask_suffix(variant)
    if pretrained == "scratch":
        return f"force_vector_scratch_mask_{suffix}"
    return f"{pretrained}_pt_force_vector_transfer_mask_{suffix}"


def run_command(cmd, dry_run=False):
    logger.info("Command: %s", " ".join(str(part) for part in cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def write_manifest(manifest_path, manifest):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def collect_sweep_summary(args, sweep_output_dir):
    rows = []
    for variant in args.mask_variants:
        variant_suffix = mask_suffix(variant)
        for pretrained in args.pretrained:
            run_name = experiment_name(pretrained, variant)
            run_output_dir = PHYSICS_OUTPUT_DIR / args.model_type / run_name
            best_path = run_output_dir / "best_metrics.json"
            history_path = run_output_dir / "metrics_history.json"
            plot_summary_path = run_output_dir / "figs" / "plot_summary.json"
            physics_metrics_path = run_output_dir / "physics_metrics" / "summary.json"

            row = {
                "mask_variant": variant_suffix,
                "pretrained": pretrained,
                "experiment_name": run_name,
                "output_dir": str(run_output_dir),
                "has_best_metrics": best_path.exists(),
                "has_metrics_history": history_path.exists(),
                "has_plots": plot_summary_path.exists(),
                "has_physics_metrics": physics_metrics_path.exists(),
            }
            if best_path.exists():
                best = json.loads(best_path.read_text())
                row.update(
                    {
                        "best_iter": best.get("iter"),
                        "best_val_loss": best.get("val_loss_mean"),
                        "best_train_loss": best.get("train_loss_mean"),
                    }
                )
            if history_path.exists():
                history = json.loads(history_path.read_text())
                if history:
                    final = history[-1]
                    row.update(
                        {
                            "final_iter": final.get("iter"),
                            "final_val_loss": final.get("val_loss_mean"),
                            "final_train_loss": final.get("train_loss_mean"),
                        }
                    )
            if physics_metrics_path.exists():
                physics_metrics = json.loads(physics_metrics_path.read_text())
                row.update(
                    {
                        "physics_component_mse": physics_metrics.get("component_mse"),
                        "physics_vector_mse": physics_metrics.get("vector_mse"),
                        "direction_cosine_mean": physics_metrics.get(
                            "direction_cosine_mean"
                        ),
                        "direction_positive_fraction": physics_metrics.get(
                            "direction_positive_fraction"
                        ),
                        "radial_alignment_mean": physics_metrics.get(
                            "radial_alignment_mean"
                        ),
                        "inward_fraction": physics_metrics.get("inward_fraction"),
                        "normalized_torque_abs_mean": physics_metrics.get(
                            "normalized_torque_abs_mean"
                        ),
                        "relative_magnitude_error_mean": physics_metrics.get(
                            "relative_magnitude_error_mean"
                        ),
                        "inverse_square_cv": physics_metrics.get("inverse_square_cv"),
                    }
                )
            rows.append(row)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mask_variants": args.mask_variants,
        "pretrained": args.pretrained,
        "rows": rows,
    }
    summary_path = sweep_output_dir / "sweep_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    csv_path = sweep_output_dir / "sweep_summary.csv"
    fieldnames = [
        "mask_variant",
        "pretrained",
        "experiment_name",
        "best_iter",
        "best_val_loss",
        "best_train_loss",
        "final_iter",
        "final_val_loss",
        "final_train_loss",
        "has_best_metrics",
        "has_metrics_history",
        "has_plots",
        "has_physics_metrics",
        "physics_component_mse",
        "physics_vector_mse",
        "direction_cosine_mean",
        "direction_positive_fraction",
        "radial_alignment_mean",
        "inward_fraction",
        "normalized_torque_abs_mean",
        "relative_magnitude_error_mean",
        "inverse_square_cv",
        "output_dir",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    logger.info("Saved sweep summary to %s and %s", summary_path, csv_path)
    return summary


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    sweep_output_dir = args.output_dir or (
        PHYSICS_OUTPUT_DIR / args.model_type / "force_mask_sweep"
    )
    sweep_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_output_dir / "sweep_manifest.json"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mask_variants": args.mask_variants,
        "pretrained": args.pretrained,
        "model_type": args.model_type,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "runs": [],
    }

    mask_files = {}
    for variant in args.mask_variants:
        path = make_mask_variant(
            variant,
            seed=args.seed,
            force=args.force_regenerate_masks,
        )
        mask_files[mask_suffix(variant)] = str(path)

    for variant in args.mask_variants:
        variant_suffix = mask_suffix(variant)
        for pretrained in args.pretrained:
            run_name = experiment_name(pretrained, variant)
            run_output_dir = PHYSICS_OUTPUT_DIR / args.model_type / run_name
            summary_path = run_output_dir / "training_summary.json"
            run_record = {
                "experiment_name": run_name,
                "pretrained": pretrained,
                "mask_variant": variant_suffix,
                "mask_file": mask_files[variant_suffix],
                "output_dir": str(run_output_dir),
                "status": "pending",
            }

            train_cmd = [
                sys.executable,
                str(script_dir / "train_model.py"),
                "--config",
                "force_vector_config",
                "--model_type",
                args.model_type,
                "--pretrained",
                pretrained,
                "--experiment_name",
                run_name,
                "--force_vector_mask_variant",
                variant_suffix,
                "--max_iters",
                str(args.max_iters),
                "--eval_interval",
                str(args.eval_interval),
                "--eval_iters",
                str(args.eval_iters),
                "--batch_size",
                str(args.batch_size),
                "--dtype",
                args.dtype,
                "--device",
                args.device,
            ]
            if not args.use_wandb:
                train_cmd.append("--no_wandb")
            if not args.compile:
                train_cmd.append("--no_compile")

            run_record["train_command"] = train_cmd

            if args.skip_training:
                run_record["status"] = "training_skipped"
            elif args.skip_existing and summary_path.exists():
                run_record["status"] = "existing_training_summary"
                logger.info("Skipping existing run %s", run_name)
            else:
                rc = run_command(train_cmd, dry_run=args.dry_run)
                run_record["train_returncode"] = rc
                run_record["status"] = "trained" if rc == 0 else "train_failed"
                if rc != 0:
                    manifest["runs"].append(run_record)
                    write_manifest(manifest_path, manifest)
                    raise SystemExit(rc)

            if not args.skip_plots:
                plot_summary_path = run_output_dir / "figs" / "plot_summary.json"
                plot_cmd = [
                    sys.executable,
                    str(script_dir / "plot_forces.py"),
                    "--model_type",
                    args.model_type,
                    "--experiment_name",
                    run_name,
                ]
                if args.skip_animation:
                    plot_cmd.append("--skip_animation")
                run_record["plot_command"] = plot_cmd
                if args.skip_existing and plot_summary_path.exists():
                    run_record["plot_status"] = "existing_plot_summary"
                    logger.info("Skipping existing plots for %s", run_name)
                elif args.dry_run:
                    run_command(plot_cmd, dry_run=True)
                    run_record["plot_status"] = "dry_run"
                else:
                    rc = run_command(plot_cmd)
                    run_record["plot_returncode"] = rc
                    run_record["plot_status"] = "plotted" if rc == 0 else "plot_failed"
                    if rc != 0:
                        manifest["runs"].append(run_record)
                        write_manifest(manifest_path, manifest)
                        raise SystemExit(rc)
            else:
                run_record["plot_status"] = "skipped"

            if not args.skip_physics_metrics:
                physics_metrics_path = run_output_dir / "physics_metrics" / "summary.json"
                metrics_cmd = [
                    sys.executable,
                    str(script_dir / "evaluate_force_physics_metrics.py"),
                    "--model_type",
                    args.model_type,
                    "--experiment_name",
                    run_name,
                    "--device",
                    args.device,
                ]
                run_record["physics_metrics_command"] = metrics_cmd
                if args.skip_existing and physics_metrics_path.exists():
                    run_record["physics_metrics_status"] = "existing_summary"
                    logger.info("Skipping existing physics metrics for %s", run_name)
                elif args.dry_run:
                    run_command(metrics_cmd, dry_run=True)
                    run_record["physics_metrics_status"] = "dry_run"
                else:
                    rc = run_command(metrics_cmd)
                    run_record["physics_metrics_returncode"] = rc
                    run_record["physics_metrics_status"] = (
                        "evaluated" if rc == 0 else "failed"
                    )
                    if rc != 0:
                        manifest["runs"].append(run_record)
                        write_manifest(manifest_path, manifest)
                        raise SystemExit(rc)
            else:
                run_record["physics_metrics_status"] = "skipped"

            manifest["runs"].append(run_record)
            write_manifest(manifest_path, manifest)

    collect_sweep_summary(args, sweep_output_dir)
    logger.info("Saved sweep manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
