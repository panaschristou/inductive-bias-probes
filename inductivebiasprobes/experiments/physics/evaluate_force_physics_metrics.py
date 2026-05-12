import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from inductivebiasprobes import Model, ModelConfig
from inductivebiasprobes.paths import (
    PHYSICS_CKPT_DIR,
    PHYSICS_CONFIG_DIR,
    PHYSICS_DATA_DIR,
    PHYSICS_OUTPUT_DIR,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PLANET_NAMES = [
    "mercury",
    "venus",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate physics-structure metrics for force-vector predictions."
    )
    parser.add_argument("--model_type", type=str, default="gpt")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--experiment_names", nargs="+", default=None)
    parser.add_argument(
        "--from_sweep_summary",
        type=Path,
        default=None,
        help="Optional sweep_summary.csv to evaluate every experiment_name listed there.",
    )
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory for a single experiment. Defaults to "
            "outputs/physics/{model_type}/{experiment_name}/physics_metrics."
        ),
    )
    parser.add_argument(
        "--aggregate_output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for aggregate summary when evaluating multiple experiments. "
            "Defaults to outputs/physics/{model_type}/force_physics_metrics."
        ),
    )
    return parser.parse_args()


def resolve_checkpoint_path(experiment_name, model_type, checkpoint_path=None):
    if checkpoint_path is not None:
        return checkpoint_path

    run_dir = PHYSICS_CKPT_DIR / model_type / experiment_name
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        return ckpt_path

    last_ckpt_path = run_dir / "last_ckpt.pt"
    if last_ckpt_path.exists():
        return last_ckpt_path

    raise FileNotFoundError(
        f"No checkpoint found for {experiment_name!r}. Expected {ckpt_path} or {last_ckpt_path}."
    )


def load_model(experiment_name, model_type="gpt", device=None, checkpoint_path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint_path(experiment_name, model_type, checkpoint_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]

    load_model_args = {}
    for key in ModelConfig.__dataclass_fields__:
        if key in checkpoint_model_args:
            load_model_args[key] = checkpoint_model_args[key]

    model_config = ModelConfig(**load_model_args)
    model = Model(model_config)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logger.warning("Missing keys while loading %s: %s", ckpt_path, incompatible.missing_keys)
    if incompatible.unexpected_keys:
        logger.warning(
            "Unexpected keys while loading %s: %s", ckpt_path, incompatible.unexpected_keys
        )
    model.config = model_config
    model.to(device)
    model.eval()
    return model, ckpt_path


def undiscretize_data(data, num_bins, min_value, max_value):
    width = (max_value - min_value) / num_bins
    return min_value + (data + 0.5) * width


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def coefficient_of_variation(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    mean = np.mean(values)
    return float(np.std(values) / (abs(mean) + eps))


def vector_metrics(pred, truth, positions, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    finite_mask = (
        np.isfinite(pred).all(axis=1)
        & np.isfinite(truth).all(axis=1)
        & np.isfinite(positions).all(axis=1)
    )
    pred = pred[finite_mask]
    truth = truth[finite_mask]
    positions = positions[finite_mask]

    if len(pred) == 0:
        return {}, {}

    diff = pred - truth
    pred_norm = np.linalg.norm(pred, axis=1)
    truth_norm = np.linalg.norm(truth, axis=1)
    r_norm = np.linalg.norm(positions, axis=1)
    inward = -positions

    nonzero_pred = pred_norm > eps
    nonzero_truth = truth_norm > eps
    nonzero_r = r_norm > eps
    valid_direction = nonzero_pred & nonzero_truth
    valid_radial = nonzero_pred & nonzero_r
    valid_true_radial = nonzero_truth & nonzero_r

    direction_cosine = np.sum(pred * truth, axis=1)[valid_direction] / (
        pred_norm[valid_direction] * truth_norm[valid_direction] + eps
    )
    radial_alignment = np.sum(pred * inward, axis=1)[valid_radial] / (
        pred_norm[valid_radial] * r_norm[valid_radial] + eps
    )
    true_radial_alignment = np.sum(truth * inward, axis=1)[valid_true_radial] / (
        truth_norm[valid_true_radial] * r_norm[valid_true_radial] + eps
    )

    torque = positions[:, 0] * pred[:, 1] - positions[:, 1] * pred[:, 0]
    true_torque = positions[:, 0] * truth[:, 1] - positions[:, 1] * truth[:, 0]
    valid_torque = nonzero_pred & nonzero_r
    normalized_torque_abs = np.abs(torque[valid_torque]) / (
        r_norm[valid_torque] * pred_norm[valid_torque] + eps
    )
    valid_true_torque = nonzero_truth & nonzero_r
    true_normalized_torque_abs = np.abs(true_torque[valid_true_torque]) / (
        r_norm[valid_true_torque] * truth_norm[valid_true_torque] + eps
    )

    inv_square = pred_norm[nonzero_r] * (r_norm[nonzero_r] ** 2)
    true_inv_square = truth_norm[nonzero_r] * (r_norm[nonzero_r] ** 2)
    rel_mag_error = np.abs(pred_norm - truth_norm) / (truth_norm + eps)

    per_timestep = {
        "component_mse_values": (diff**2).reshape(-1),
        "vector_squared_error_values": np.sum(diff**2, axis=1),
        "direction_cosine_values": direction_cosine,
        "radial_alignment_values": radial_alignment,
        "true_radial_alignment_values": true_radial_alignment,
        "normalized_torque_abs_values": normalized_torque_abs,
        "true_normalized_torque_abs_values": true_normalized_torque_abs,
        "relative_magnitude_error_values": rel_mag_error,
        "absolute_magnitude_error_values": np.abs(pred_norm - truth_norm),
        "inverse_square_values": inv_square,
        "true_inverse_square_values": true_inv_square,
    }

    summary = {
        "num_points": int(len(pred)),
        "component_mse": safe_mean(per_timestep["component_mse_values"]),
        "vector_mse": safe_mean(per_timestep["vector_squared_error_values"]),
        "direction_cosine_mean": safe_mean(direction_cosine),
        "direction_cosine_median": float(np.median(direction_cosine))
        if direction_cosine.size
        else float("nan"),
        "direction_positive_fraction": safe_mean(direction_cosine > 0),
        "radial_alignment_mean": safe_mean(radial_alignment),
        "radial_alignment_median": float(np.median(radial_alignment))
        if radial_alignment.size
        else float("nan"),
        "inward_fraction": safe_mean(radial_alignment > 0),
        "true_radial_alignment_mean": safe_mean(true_radial_alignment),
        "normalized_torque_abs_mean": safe_mean(normalized_torque_abs),
        "normalized_torque_abs_median": float(np.median(normalized_torque_abs))
        if normalized_torque_abs.size
        else float("nan"),
        "true_normalized_torque_abs_mean": safe_mean(true_normalized_torque_abs),
        "relative_magnitude_error_mean": safe_mean(rel_mag_error),
        "absolute_magnitude_error_mean": safe_mean(np.abs(pred_norm - truth_norm)),
        "predicted_magnitude_mean": safe_mean(pred_norm),
        "true_magnitude_mean": safe_mean(truth_norm),
        "inverse_square_mean": safe_mean(inv_square),
        "inverse_square_std": safe_std(inv_square),
        "inverse_square_cv": coefficient_of_variation(inv_square),
        "true_inverse_square_cv": coefficient_of_variation(true_inv_square),
    }
    return summary, per_timestep


def predict_solar_system_forces(model, device):
    with (PHYSICS_CONFIG_DIR / "force_vector_config.yaml").open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    obs = np.load(PHYSICS_DATA_DIR / "obs_solar_system_two_body.npy")
    true_force_vectors = np.load(PHYSICS_DATA_DIR / "force_vector_solar_system_two_body.npy")

    all_preds = []
    all_truth = []
    all_positions = []
    for planet_idx in range(obs.shape[0]):
        x = torch.from_numpy(obs[planet_idx]).long().unsqueeze(0)[:, :-1, :].to(device)
        with torch.no_grad():
            output = model(x)

        pred = output[0, :, :2].detach().cpu().numpy()
        truth = true_force_vectors[planet_idx][:-1, :2]
        positions = undiscretize_data(
            obs[planet_idx, :-1, :],
            config["input_vocab_size"] - 3,
            -50,
            50,
        )

        all_preds.append(pred[1::2, :])
        all_truth.append(truth[1::2, :])
        all_positions.append(positions[1::2, :])

    return all_preds, all_truth, all_positions


def evaluate_experiment(experiment_name, model_type="gpt", device=None, output_dir=None, checkpoint_path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_path = load_model(
        experiment_name,
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    all_preds, all_truth, all_positions = predict_solar_system_forces(model, device)

    by_planet = []
    aggregate_values = {}
    for planet_name, pred, truth, positions in zip(
        PLANET_NAMES, all_preds, all_truth, all_positions
    ):
        summary, values = vector_metrics(pred, truth, positions)
        summary["planet"] = planet_name
        by_planet.append(summary)
        for key, value in values.items():
            aggregate_values.setdefault(key, []).append(value)

    flattened_values = {
        key: np.concatenate(value) if value else np.array([])
        for key, value in aggregate_values.items()
    }
    aggregate_summary, _ = vector_metrics(
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_truth, axis=0),
        np.concatenate(all_positions, axis=0),
    )
    aggregate_summary.update(
        {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "checkpoint_path": str(ckpt_path),
            "device": device,
        }
    )
    # Use the exact flattened values for inverse-square aggregate CV fields.
    aggregate_summary["inverse_square_cv"] = coefficient_of_variation(
        flattened_values.get("inverse_square_values", np.array([]))
    )
    aggregate_summary["true_inverse_square_cv"] = coefficient_of_variation(
        flattened_values.get("true_inverse_square_values", np.array([]))
    )

    output_dir = output_dir or (
        PHYSICS_OUTPUT_DIR / model_type / experiment_name / "physics_metrics"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "summary.json").open("w") as f:
        json.dump(aggregate_summary, f, indent=2)
    with (output_dir / "by_planet.json").open("w") as f:
        json.dump(by_planet, f, indent=2)

    fieldnames = [
        "planet",
        "num_points",
        "component_mse",
        "vector_mse",
        "direction_cosine_mean",
        "direction_cosine_median",
        "direction_positive_fraction",
        "radial_alignment_mean",
        "radial_alignment_median",
        "inward_fraction",
        "normalized_torque_abs_mean",
        "normalized_torque_abs_median",
        "relative_magnitude_error_mean",
        "absolute_magnitude_error_mean",
        "predicted_magnitude_mean",
        "true_magnitude_mean",
        "inverse_square_cv",
        "true_inverse_square_cv",
    ]
    with (output_dir / "by_planet.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_planet:
            writer.writerow({key: row.get(key) for key in fieldnames})

    logger.info("Saved physics metrics for %s to %s", experiment_name, output_dir)
    return aggregate_summary


def experiment_names_from_args(args):
    names = []
    if args.experiment_name:
        names.append(args.experiment_name)
    if args.experiment_names:
        names.extend(args.experiment_names)
    if args.from_sweep_summary is not None:
        with args.from_sweep_summary.open("r", newline="") as f:
            for row in csv.DictReader(f):
                names.append(row["experiment_name"])
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(names))


def write_aggregate_summary(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "physics_metrics_summary.json").open("w") as f:
        json.dump({"runs": rows}, f, indent=2)

    fieldnames = [
        "experiment_name",
        "model_type",
        "component_mse",
        "vector_mse",
        "direction_cosine_mean",
        "direction_positive_fraction",
        "radial_alignment_mean",
        "inward_fraction",
        "normalized_torque_abs_mean",
        "relative_magnitude_error_mean",
        "absolute_magnitude_error_mean",
        "inverse_square_cv",
        "true_inverse_square_cv",
        "checkpoint_path",
    ]
    with (output_dir / "physics_metrics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    names = experiment_names_from_args(args)
    if not names:
        raise ValueError(
            "Provide --experiment_name, --experiment_names, or --from_sweep_summary."
        )

    rows = []
    for name in names:
        output_dir = args.output_dir if len(names) == 1 else None
        rows.append(
            evaluate_experiment(
                name,
                model_type=args.model_type,
                device=args.device,
                output_dir=output_dir,
                checkpoint_path=args.checkpoint_path if len(names) == 1 else None,
            )
        )

    aggregate_output_dir = args.aggregate_output_dir or (
        PHYSICS_OUTPUT_DIR / args.model_type / "force_physics_metrics"
    )
    write_aggregate_summary(rows, Path(aggregate_output_dir))


if __name__ == "__main__":
    main()
