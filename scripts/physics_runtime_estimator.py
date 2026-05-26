#!/usr/bin/env python3
"""Estimate storage and rough wall time for full physics experiments.

The generator default in inductivebiasprobes/experiments/physics/generate_data.py
is 10,000,000 train trajectories. This script makes the cost visible before
launching a long NERSC job.
"""

from __future__ import annotations

import argparse


BYTES_PER_GIB = 1024**3


def gib(num_bytes: float) -> float:
    return num_bytes / BYTES_PER_GIB


def estimate_dataset_bytes(
    trajectories: int,
    points_per_trajectory: int,
    save_pretraining_forces: bool,
) -> dict[str, float]:
    """Return rough bytes for the dominant train split files."""
    # With num_bins > 0 the observation memmap is uint16:
    # shape = (trajectories, points, 2)
    obs = trajectories * points_per_trajectory * 2 * 2
    num_bodies = trajectories

    parts = {
        "obs_train_uint16": obs,
        "num_bodies_train_uint8": num_bodies,
    }

    if save_pretraining_forces:
        # Optional train split physics targets:
        # state: 5 float32, full_state: 6 float32, force_vector: 2 float32,
        # force_magnitude: 1 float32, hamiltonian: 1 float32,
        # angular_momentum: 1 float32.
        physics_dims = {
            "state_train_float32": 5,
            "full_state_train_float32": 6,
            "force_vector_train_float32": 2,
            "force_magnitude_train_float32": 1,
            "hamiltonian_train_float32": 1,
            "angular_momentum_train_float32": 1,
        }
        for name, dim in physics_dims.items():
            parts[name] = trajectories * points_per_trajectory * dim * 4

    return parts


def estimate_wall_time_hours(args: argparse.Namespace) -> float | None:
    if args.observed_hours is None:
        return None

    trajectory_scale = args.full_train_trajectories / args.observed_trajectories
    if args.target_workers <= 0 or args.observed_workers <= 0:
        raise ValueError("Worker counts must be positive.")
    if args.parallel_efficiency <= 0:
        raise ValueError("parallel_efficiency must be positive.")

    worker_scale = args.observed_workers / (
        args.target_workers * args.parallel_efficiency
    )
    return args.observed_hours * trajectory_scale * worker_scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate full physics dataset storage and rough runtime."
    )
    parser.add_argument("--full_train_trajectories", type=int, default=10_000_000)
    parser.add_argument("--points_per_trajectory", type=int, default=1_000)
    parser.add_argument(
        "--save_pretraining_forces",
        action="store_true",
        help="Include optional train split state/force/H/L targets in storage estimate.",
    )
    parser.add_argument(
        "--observed_hours",
        type=float,
        default=None,
        help=(
            "Observed runtime to scale from. If your full generation attempt said "
            "30 hours, pass --observed_hours 30."
        ),
    )
    parser.add_argument(
        "--observed_trajectories",
        type=int,
        default=10_000_000,
        help="Trajectory count used in the observed runtime.",
    )
    parser.add_argument("--observed_workers", type=int, default=90)
    parser.add_argument("--target_workers", type=int, default=120)
    parser.add_argument(
        "--parallel_efficiency",
        type=float,
        default=0.85,
        help=(
            "Penalty for imperfect scaling when changing worker counts. "
            "Use 1.0 for ideal scaling."
        ),
    )
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_iters", type=int, default=600_000)
    args = parser.parse_args()

    parts = estimate_dataset_bytes(
        args.full_train_trajectories,
        args.points_per_trajectory,
        args.save_pretraining_forces,
    )
    total = sum(parts.values())

    print("Full physics scale estimate")
    print("===========================")
    print(f"train trajectories: {args.full_train_trajectories:,}")
    print(f"points per trajectory: {args.points_per_trajectory:,}")
    print()
    print("Dominant train split storage:")
    for name, size in parts.items():
        print(f"  {name:32s} {gib(size):10.2f} GiB")
    print(f"  {'total listed':32s} {gib(total):10.2f} GiB")
    print()
    print(
        "Note: val/test/traj/two_body files add a small amount compared with "
        "the 10M train split unless --save_pretraining_forces is used."
    )
    print()

    wall_time = estimate_wall_time_hours(args)
    if wall_time is None:
        print(
            "Runtime: pass --observed_hours to scale from a real NERSC timing, "
            "for example --observed_hours 30."
        )
    else:
        print(
            "Estimated generation wall time at "
            f"{args.target_workers} workers: {wall_time:.2f} hours"
        )
        print(
            "This is a rough linear estimate. Queue placement, filesystem load, "
            "and the two internal generation phases can move it around."
        )
    print()

    effective_batch = (
        args.batch_size * args.gradient_accumulation_steps * args.num_gpus
    )
    iters_per_epoch = max(args.full_train_trajectories // effective_batch, 1)
    epochs = args.max_iters / iters_per_epoch
    print("Training schedule shape:")
    print(f"  num_gpus: {args.num_gpus}")
    print(f"  batch_size per GPU: {args.batch_size}")
    print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"  effective batch: {effective_batch:,} trajectories")
    print(f"  iters per 10M epoch: {iters_per_epoch:,}")
    print(f"  epochs in {args.max_iters:,} iterations: {epochs:.2f}")


if __name__ == "__main__":
    main()
