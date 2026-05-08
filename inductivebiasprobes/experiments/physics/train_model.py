import argparse
import logging
import yaml
import wandb

from inductivebiasprobes.paths import (
    PHYSICS_CONFIG_DIR,
    PHYSICS_CKPT_DIR,
    PHYSICS_DATA_DIR,
    PHYSICS_EXT_DIR,
)
from inductivebiasprobes.src.train_utils import (
    add_common_args,
    generate_and_save_extrapolations,
    init_model,
    setup_training_environment,
    train,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_physics_args():
    """Parse physics-specific command line arguments."""
    parser = argparse.ArgumentParser(description="Train a physics model")
    parser = add_common_args(parser)
    parser.add_argument("--white_noise_dataset_idx_lower", type=int, default=None)
    parser.add_argument("--white_noise_dataset_idx_upper", type=int, default=None)
    parser.add_argument(
        "--force_aux_dataset",
        type=str,
        default=None,
        choices=["pretraining", "two_body"],
        help="Dataset used for force-aware next-token pretraining",
    )
    return parser.parse_args()


def load_config(args):
    """Load configuration from file and command line args."""
    config = vars(args)
    assert args.config is not None, "Config file is required"
    with (PHYSICS_CONFIG_DIR / (args.config + ".yaml")).open("r") as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in config.items():
        if value is not None:
            file_config[key] = value
    return file_config


def _require_file(path, config_name):
    if not path.exists():
        raise FileNotFoundError(
            f"{config_name} was set to {path}, but that file does not exist. "
            "Regenerate data with the needed force/state targets or use "
            "--force_aux_dataset two_body for the first prototype."
        )


def configure_force_auxiliary_files(config):
    """Attach force/state target files for force-aware physics pretraining."""
    if not (
        config.get("use_force_aux_loss")
        or config.get("use_force_law_loss")
        or config.get("use_hamiltonian_aux_loss")
        or config.get("use_angular_momentum_aux_loss")
    ):
        return
    if config["predict_type"] != "next_token":
        raise ValueError(
            "Force auxiliary/law losses are currently wired for next_token "
            "pretraining. Use transfer configs without these flags."
        )

    dataset = config.get("force_aux_dataset", "pretraining")
    if dataset == "two_body":
        train_label, val_label = "two_body_train", "two_body_val"
        config["train_file"] = PHYSICS_DATA_DIR / f"obs_{train_label}.npy"
        config["val_file"] = PHYSICS_DATA_DIR / f"obs_{val_label}.npy"
        config["test_file"] = PHYSICS_DATA_DIR / "obs_two_body_test.npy"
        config["force_aux_shift_targets"] = True
        config["force_law_shift_targets"] = True
        config["hamiltonian_aux_shift_targets"] = True
        config["angular_momentum_aux_shift_targets"] = True
        logger.info("Using two-body data for force-aware next-token pretraining")
    elif dataset == "pretraining":
        train_label, val_label = "train", "val"
        logger.info("Using ordinary pretraining data for force-aware pretraining")
    else:
        raise ValueError(f"Unsupported force_aux_dataset: {dataset}")

    if config.get("use_force_aux_loss"):
        train_force = PHYSICS_DATA_DIR / f"force_vector_{train_label}.npy"
        val_force = PHYSICS_DATA_DIR / f"force_vector_{val_label}.npy"
        _require_file(train_force, "train_force_aux_target_file")
        _require_file(val_force, "val_force_aux_target_file")
        config["train_force_aux_target_file"] = train_force
        config["val_force_aux_target_file"] = val_force
        config.setdefault("force_aux_dim", 2)
        config.setdefault("force_aux_mask_id", float("inf"))

    if config.get("use_force_law_loss"):
        train_full_state = PHYSICS_DATA_DIR / f"full_state_{train_label}.npy"
        val_full_state = PHYSICS_DATA_DIR / f"full_state_{val_label}.npy"
        _require_file(train_full_state, "train_full_state_file")
        _require_file(val_full_state, "val_full_state_file")
        config["train_full_state_file"] = train_full_state
        config["val_full_state_file"] = val_full_state
        config.setdefault("force_aux_dim", 2)

    if config.get("use_hamiltonian_aux_loss"):
        train_hamiltonian = PHYSICS_DATA_DIR / f"hamiltonian_{train_label}.npy"
        val_hamiltonian = PHYSICS_DATA_DIR / f"hamiltonian_{val_label}.npy"
        _require_file(train_hamiltonian, "train_hamiltonian_aux_target_file")
        _require_file(val_hamiltonian, "val_hamiltonian_aux_target_file")
        config["train_hamiltonian_aux_target_file"] = train_hamiltonian
        config["val_hamiltonian_aux_target_file"] = val_hamiltonian
        config.setdefault("hamiltonian_aux_mask_id", float("inf"))

    if config.get("use_angular_momentum_aux_loss"):
        train_angular = PHYSICS_DATA_DIR / f"angular_momentum_{train_label}.npy"
        val_angular = PHYSICS_DATA_DIR / f"angular_momentum_{val_label}.npy"
        _require_file(train_angular, "train_angular_momentum_aux_target_file")
        _require_file(val_angular, "val_angular_momentum_aux_target_file")
        config["train_angular_momentum_aux_target_file"] = train_angular
        config["val_angular_momentum_aux_target_file"] = val_angular
        config.setdefault("angular_momentum_aux_mask_id", float("inf"))


def train_and_save_model(
    config,
    pretrained_ckpt_dir,
    save_ckpt_dir,
    run_name=None,
    white_noise_dataset_idx=None,
):
    """Initialize, train and optionally save model predictions."""
    save_checkpoints = config["predict_type"] != "white_noise"
    # Setup training environment
    ddp, master_process, ptdtype, config = setup_training_environment(
        config, save_ckpt_dir, save_checkpoints
    )

    # Create dataloaders
    config["train_file"] = PHYSICS_DATA_DIR / "obs_train.npy"
    config["val_file"] = PHYSICS_DATA_DIR / "obs_val.npy"
    config["test_file"] = PHYSICS_DATA_DIR / "obs_test.npy"
    config["use_float_x"] = False 
    config["use_float_y"] = config["output_vocab_size"] is None
    train_label, val_label = "train", "val"
    if config["predict_type"] == "state":
        config["train_target_file"] = (
            PHYSICS_DATA_DIR / f"{config['predict_type']}_train.npy"
        )
        config["val_target_file"] = (
            PHYSICS_DATA_DIR / f"{config['predict_type']}_val.npy"
        )
    elif "force" in config["predict_type"]:
        if config["predict_type"] == "force_magnitude":
            print("NOTE: USING MASKED DATA")
            config["train_file"] = PHYSICS_DATA_DIR / f"obs_two_body_train.npy"
            config["val_file"] = PHYSICS_DATA_DIR / f"obs_two_body_train.npy"
            config["train_target_file"] = (
                PHYSICS_DATA_DIR / f"force_magnitude_two_body_train_masked.npy"
            )
            config["val_target_file"] = (
                PHYSICS_DATA_DIR / f"force_magnitude_two_body_train.npy"
            )
        elif config["predict_type"] == "force_vector":
            print("NOTE: USING MASKED SOLAR SYSTEM DATA")
            config['train_file'] = PHYSICS_DATA_DIR / f"obs_solar_system_two_body.npy"
            config['val_file'] = PHYSICS_DATA_DIR / f"obs_solar_system_two_body.npy"
            config["train_target_file"] = (
                PHYSICS_DATA_DIR / f"force_vector_solar_system_two_body_masked.npy"
            )
            config["val_target_file"] = (
                PHYSICS_DATA_DIR / f"force_vector_solar_system_two_body.npy"
            )
    elif config["predict_type"] == "white_noise":
        config["train_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_obs_train_{white_noise_dataset_idx}.npy"
        )
        config["val_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_obs_val_{white_noise_dataset_idx}.npy"
        )
        config["train_target_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_output_train_{white_noise_dataset_idx}.npy"
        )
        config["val_target_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_states_val_{white_noise_dataset_idx}.npy"
        )
        config["train_indices_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_indices_train_{white_noise_dataset_idx}.npy"
        )
        config["val_indices_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_indices_val_{white_noise_dataset_idx}.npy"
        )
    configure_force_auxiliary_files(config)

    # Setup wandb config
    config["wandb_project"] = f"physics-pretrain-{config['predict_type']}"
    config["wandb_entity"] = "petergchang"
    # Override only if run_name is not provided
    if config["wandb_run_name"] == "default":
        config["wandb_run_name"] = run_name or "gpt"
    if config["predict_type"] == "white_noise":
        config["no_wandb"] = True

    # Set target callback
    target_callback = None
    loss_name = None

    # Initialize model
    model, config, iter_num, current_epoch, best_val_loss, optimizer, scaler = (
        init_model(
            config=config,
            ckpt_dir=pretrained_ckpt_dir,
            ddp=ddp,
        )
    )

    # Setup wandb logging
    if not config["no_wandb"] and master_process:
        wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=config["wandb_run_name"],
            resume="allow",
            config=config,
        )

    # Train model
    train(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        ddp=ddp,
        master_process=master_process,
        ptdtype=ptdtype,
        iter_num=iter_num,
        current_epoch=current_epoch,
        best_val_loss=best_val_loss,
        ckpt_dir=save_ckpt_dir,
        save_checkpoints=save_checkpoints,
        target_callback=target_callback,
        loss_name=loss_name,
    )

    # For white noise models or acceleration magnitude, generate and save extrapolations
    if config["predict_type"] == "white_noise":
        ext_dir = PHYSICS_EXT_DIR / config["predict_type"]
        ext_idx_dir = (
            ext_dir
            / config["model_type"]
            / f"pt_{config['pretrained']}"
            / f"{config['white_noise_dataset_size']}_examples"
            / f"{config['max_iters']}_iters"
            / f"idx_{white_noise_dataset_idx}"
        )
        generate_and_save_extrapolations(model, config, ext_dir, ext_idx_dir)

    if not config["no_wandb"] and master_process:
        wandb.finish()

    return model


def main():
    # Parse arguments and load config
    args = parse_physics_args()
    config = load_config(args)

    save_name = config.get("experiment_name") or config["predict_type"]

    # Setup pretrained checkpoint directory. Default is scratch.
    if config["pretrained"] == "scratch":
        pretrained_ckpt_dir = (
            PHYSICS_CKPT_DIR / config["model_type"] / save_name
        )
    else:
        pretrained_ckpt_dir = (
            PHYSICS_CKPT_DIR / config["model_type"] / config["pretrained"]
        )

    if config["predict_type"] == "white_noise":
        if config["white_noise_dataset_idx_lower"] is None or config["white_noise_dataset_idx_upper"] is None:
            idx_range = range(config["num_white_noise_datasets"])
        else:
            idx_range = range(config["white_noise_dataset_idx_lower"], config["white_noise_dataset_idx_upper"])
        for dataset_idx in idx_range:
            logger.info(f"[Training on white noise dataset {dataset_idx}]")

            # Build checkpoint path
            ckpt_name = f"{config['pretrained']}_pt_{config['predict_type']}"
            ckpt_name += f"_idx_{dataset_idx}_transfer"

            save_ckpt_dir = PHYSICS_CKPT_DIR / config["model_type"] / ckpt_name

            # Build run name
            run_name = (
                f"{config['white_noise_dataset_size']}_examples_"
                f"{config['max_iters']}_iters_batch_{dataset_idx}"
            )

            train_and_save_model(
                config,
                pretrained_ckpt_dir,
                save_ckpt_dir,
                run_name=run_name,
                white_noise_dataset_idx=dataset_idx,
            )
    else:
        # Setup checkpoint directories
        if config["pretrained"] == "scratch":
            save_ckpt_dir = (
                PHYSICS_CKPT_DIR / config["model_type"] / save_name
            )
        else:
            transfer_name = config.get("experiment_name")
            if transfer_name is None:
                transfer_name = f"{config['pretrained']}_pt_{config['predict_type']}_transfer"
            save_ckpt_dir = (
                PHYSICS_CKPT_DIR
                / config["model_type"]
                / transfer_name
            )

        train_and_save_model(
            config,
            pretrained_ckpt_dir,
            save_ckpt_dir,
            run_name=f'pt_{config["pretrained"]}_{config["model_type"]}',
        )


if __name__ == "__main__":
    main()
