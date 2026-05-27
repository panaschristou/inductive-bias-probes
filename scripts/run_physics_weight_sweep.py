#!/usr/bin/env python3
"""Run a force-law / force-aux loss-weight sweep and collect results.

This script is intended to be run from the repository root:

    python scripts/run_physics_weight_sweep.py

It sequentially:
1. pretrains force-law and force-aux variants at each requested loss weight;
2. runs force-vector transfer on the requested mask variants;
3. runs plots and physics metrics through run_force_mask_sweep.py;
4. writes a compact CSV/JSON summary under outputs/physics/gpt.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_DIR = REPO_ROOT / "inductivebiasprobes" / "experiments" / "physics"
OUTPUT_ROOT = REPO_ROOT / "inductivebiasprobes" / "outputs" / "physics" / "gpt"


PHYSICS_FIELDS = [
    "component_mse",
    "vector_mse",
    "direction_cosine_mean",
    "direction_positive_fraction",
    "radial_alignment_mean",
    "inward_fraction",
    "normalized_torque_abs_mean",
    "relative_magnitude_error_mean",
    "inverse_square_cv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep physics loss weights for 100k-scale force-aware pretraining."
    )
    parser.add_argument("--scale_tag", default="100k")
    parser.add_argument("--num_data_points", type=int, default=100_000)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.1],
        help="Loss weights to sweep.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=["force_law", "force_aux"],
        default=["force_law", "force_aux"],
    )
    parser.add_argument("--mask_variants", nargs="+", default=["25"])
    parser.add_argument("--pretrain_max_iters", type=int, default=6_250)
    parser.add_argument("--pretrain_eval_interval", type=int, default=100)
    parser.add_argument("--transfer_max_iters", type=int, default=10_000)
    parser.add_argument("--transfer_eval_interval", type=int, default=50)
    parser.add_argument("--eval_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_type", default="gpt")
    parser.add_argument(
        "--experiment_suffix",
        default="_weight_sweep",
        help="Suffix appended to transfer experiment names.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Summary directory. Defaults to "
            "inductivebiasprobes/outputs/physics/gpt/force_weight_sweep_{scale_tag}."
        ),
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Enable torch.compile. Default is disabled.",
    )
    parser.add_argument(
        "--no_compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile.",
    )
    parser.set_defaults(compile=False)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_pretraining", action="store_true")
    parser.add_argument("--skip_transfer", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--skip_physics_metrics", action="store_true")
    parser.add_argument("--skip_animation", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--include_baselines",
        action="store_true",
        help="Include already-existing scratch/next-token/default-weight rows in summary.",
    )
    return parser.parse_args()


def weight_tag(weight: float) -> str:
    return "w" + f"{weight:g}".replace("-", "m").replace(".", "p")


def run_name(objective: str, scale_tag: str, weight: float) -> str:
    return f"next_token_{objective}_{scale_tag}_{weight_tag(weight)}"


def transfer_name(pretrained: str, mask_variant: str, suffix: str) -> str:
    if pretrained == "scratch":
        return f"force_vector_scratch_mask_{mask_variant}{suffix}"
    return f"{pretrained}_pt_force_vector_transfer_mask_{mask_variant}{suffix}"


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool = False) -> int:
    print(">>>", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def pretrain_summary_path(name: str) -> Path:
    return OUTPUT_ROOT / name / "training_summary.json"


def transfer_summary_path(name: str) -> Path:
    return OUTPUT_ROOT / name / "training_summary.json"


def build_pretrain_command(args: argparse.Namespace, objective: str, weight: float) -> list[str]:
    if objective == "force_law":
        config = "ntp_force_law_config"
        weight_flag = "--force_law_loss_weight"
    elif objective == "force_aux":
        config = "ntp_force_aux_config"
        weight_flag = "--force_loss_weight"
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    cmd = [
        sys.executable,
        "train_model.py",
        "--config",
        config,
        "--experiment_name",
        run_name(objective, args.scale_tag, weight),
        "--num_data_points",
        str(args.num_data_points),
        weight_flag,
        str(weight),
        "--max_iters",
        str(args.pretrain_max_iters),
        "--eval_interval",
        str(args.pretrain_eval_interval),
        "--eval_iters",
        str(args.eval_iters),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
    ]
    if not args.use_wandb:
        cmd.append("--no_wandb")
    if not args.compile:
        cmd.append("--no_compile")
    return cmd


def build_transfer_command(args: argparse.Namespace, pretrained_runs: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "run_force_mask_sweep.py",
        "--mask_variants",
        *args.mask_variants,
        "--pretrained",
        *pretrained_runs,
        "--experiment_suffix",
        args.experiment_suffix,
        "--max_iters",
        str(args.transfer_max_iters),
        "--eval_interval",
        str(args.transfer_eval_interval),
        "--eval_iters",
        str(args.eval_iters),
        "--batch_size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--output_dir",
        str(args.output_dir),
    ]
    if not args.use_wandb:
        # run_force_mask_sweep defaults to no wandb unless --use_wandb is passed.
        pass
    else:
        cmd.append("--use_wandb")
    if not args.compile:
        cmd.append("--no_compile")
    else:
        cmd.append("--compile")
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.skip_plots:
        cmd.append("--skip_plots")
    if args.skip_physics_metrics:
        cmd.append("--skip_physics_metrics")
    if args.skip_animation:
        cmd.append("--skip_animation")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def collect_rows(args: argparse.Namespace, specs: list[tuple[str, float, str]]) -> list[dict]:
    rows: list[dict] = []

    for objective, weight, pretrained in specs:
        pretrain_best = read_json(OUTPUT_ROOT / pretrained / "best_metrics.json")
        for mask_variant in args.mask_variants:
            transfer = transfer_name(pretrained, mask_variant, args.experiment_suffix)
            transfer_best = read_json(OUTPUT_ROOT / transfer / "best_metrics.json")
            physics = read_json(OUTPUT_ROOT / transfer / "physics_metrics" / "summary.json")
            row = {
                "objective": objective,
                "weight": weight,
                "weight_tag": weight_tag(weight),
                "scale_tag": args.scale_tag,
                "mask_variant": mask_variant,
                "pretrained_run": pretrained,
                "transfer_run": transfer,
                "pretrain_best_iter": pretrain_best.get("iter"),
                "pretrain_best_val_loss": pretrain_best.get("val_loss_mean"),
                "pretrain_best_train_loss": pretrain_best.get("train_loss_mean"),
                "transfer_best_iter": transfer_best.get("iter"),
                "transfer_best_val_loss": transfer_best.get("val_loss_mean"),
                "transfer_best_train_loss": transfer_best.get("train_loss_mean"),
                "has_pretrain": bool(pretrain_best),
                "has_transfer": bool(transfer_best),
                "has_physics_metrics": bool(physics),
            }
            for field in PHYSICS_FIELDS:
                row[field] = physics.get(field)
            rows.append(row)

    if args.include_baselines:
        baseline_specs = [
            ("scratch", "", "scratch", f"force_vector_scratch_mask_25_{args.scale_tag}"),
            (
                "next_token",
                "",
                f"next_token_{args.scale_tag}",
                f"next_token_{args.scale_tag}_pt_force_vector_transfer_mask_25_{args.scale_tag}",
            ),
            (
                "force_law_default",
                0.1,
                f"next_token_force_law_{args.scale_tag}",
                f"next_token_force_law_{args.scale_tag}_pt_force_vector_transfer_mask_25_{args.scale_tag}",
            ),
            (
                "force_aux_default",
                0.1,
                f"next_token_force_aux_{args.scale_tag}",
                f"next_token_force_aux_{args.scale_tag}_pt_force_vector_transfer_mask_25_{args.scale_tag}",
            ),
        ]
        for objective, weight, pretrained, transfer in baseline_specs:
            pretrain_best = read_json(OUTPUT_ROOT / pretrained / "best_metrics.json")
            transfer_best = read_json(OUTPUT_ROOT / transfer / "best_metrics.json")
            physics = read_json(OUTPUT_ROOT / transfer / "physics_metrics" / "summary.json")
            row = {
                "objective": objective,
                "weight": weight,
                "weight_tag": "",
                "scale_tag": args.scale_tag,
                "mask_variant": "25",
                "pretrained_run": pretrained,
                "transfer_run": transfer,
                "pretrain_best_iter": pretrain_best.get("iter"),
                "pretrain_best_val_loss": pretrain_best.get("val_loss_mean"),
                "pretrain_best_train_loss": pretrain_best.get("train_loss_mean"),
                "transfer_best_iter": transfer_best.get("iter"),
                "transfer_best_val_loss": transfer_best.get("val_loss_mean"),
                "transfer_best_train_loss": transfer_best.get("train_loss_mean"),
                "has_pretrain": bool(pretrain_best) or objective == "scratch",
                "has_transfer": bool(transfer_best),
                "has_physics_metrics": bool(physics),
            }
            for field in PHYSICS_FIELDS:
                row[field] = physics.get(field)
            rows.append(row)

    return rows


def write_summary(args: argparse.Namespace, rows: list[dict], specs: list[tuple[str, float, str]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "objective",
        "weight",
        "weight_tag",
        "scale_tag",
        "mask_variant",
        "pretrained_run",
        "transfer_run",
        "pretrain_best_iter",
        "pretrain_best_val_loss",
        "pretrain_best_train_loss",
        "transfer_best_iter",
        "transfer_best_val_loss",
        "transfer_best_train_loss",
        "has_pretrain",
        "has_transfer",
        "has_physics_metrics",
        *PHYSICS_FIELDS,
    ]

    csv_path = args.output_dir / "weight_sweep_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scale_tag": args.scale_tag,
        "num_data_points": args.num_data_points,
        "weights": args.weights,
        "objectives": args.objectives,
        "mask_variants": args.mask_variants,
        "pretrained_runs": [
            {"objective": objective, "weight": weight, "run": run}
            for objective, weight, run in specs
        ],
        "rows": rows,
    }
    json_path = args.output_dir / "weight_sweep_summary.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"[summary] {csv_path}")
    print(f"[summary] {json_path}")


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = OUTPUT_ROOT / f"force_weight_sweep_{args.scale_tag}"
    else:
        args.output_dir = args.output_dir.resolve()

    specs: list[tuple[str, float, str]] = []
    for objective in args.objectives:
        for weight in args.weights:
            specs.append((objective, weight, run_name(objective, args.scale_tag, weight)))

    if not args.skip_pretraining:
        for objective, weight, pretrained in specs:
            summary_path = pretrain_summary_path(pretrained)
            if args.skip_existing and summary_path.exists():
                print(f"[skip] existing pretraining summary: {summary_path}")
                continue
            rc = run_command(
                build_pretrain_command(args, objective, weight),
                cwd=PHYSICS_DIR,
                dry_run=args.dry_run,
            )
            if rc != 0:
                raise SystemExit(rc)

    if not args.skip_transfer:
        pretrained_runs = [run for _, _, run in specs]
        rc = run_command(
            build_transfer_command(args, pretrained_runs),
            cwd=PHYSICS_DIR,
            dry_run=False,
        )
        if rc != 0:
            raise SystemExit(rc)

    rows = collect_rows(args, specs)
    write_summary(args, rows, specs)


if __name__ == "__main__":
    main()
