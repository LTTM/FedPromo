"""
Main script to run the simulation

Classes:
    None

Functions:
    main: Main function to run the simulation.

Constants:
    None

Exceptions:
    None

Authors: Matteo Caligiuri
         Francesco Barbato
"""

import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from datetime import datetime as date

# Set higher logging level for Hydra
os.environ["HYDRA_FULL_ERROR"] = "1"  # pylint: disable=wrong-import-position

# Set ray in debug mode
os.environ["RAY_DEBUG"] = "0"  # pylint: disable=wrong-import-position

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from torch.utils.tensorboard.writer import SummaryWriter

from modules import (
    set_device,
    set_seed,
    Pretraining,
    FedTrainer,
    Logger,
    parse_def_conf,
    get_dts,
    flatten_conf_dict,
)


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()


# Set the warnings to be ignored
warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the simulation.

    Args:
        cfg (DictConfig): The configuration object.
    """

    ## Print general information about the simulation (Date number of available gpus etc)
    logger.info("Starting simulation %s", date.now())

    ## Get experiment output dir
    save_path = Path(HydraConfig.get().runtime.output_dir)
    logger.info("Experiment will be saved in: %s", save_path)

    ## Setup logging (if present use MLFlow otherwise use tensorboard)
    if cfg.mlflow is not None:
        run_name = instantiate(cfg.mlflow)
        if run_name is None:
            # Get client model name
            client_function = getattr(
                import_module(
                    cfg.client_model._target_.split(  # pylint: disable=protected-access
                        "."
                    )[0]
                ),
                cfg.client_model._target_.split(  # pylint: disable=protected-access
                    "."
                )[1],
            )
            client_module = sys.modules[client_function.__module__]
            client_name = (
                client_module._target_class.__name__  # pylint: disable=protected-access
            )

            # Define the default run name
            run_name = (
                f"{client_name}"
                + f"_{cfg.federated.dts.name}"
                + f"_{date.now():%d-%m-%Y@%H:%M:%S}"
            )

        # Extract tge full command executed
        full_command = " ".join([sys.executable] + sys.argv)

        # Extarct the configuration parameters
        # Flatten the configuration dictionary to a single level
        # This is useful to log the configuration parameters in MLFlow
        # and to avoid nested dictionaries
        config_params = flatten_conf_dict(cfg)

        data_logger = {
            "mlflow": {
                "run_name": run_name,
                "run_id": None,
                "executed_command": full_command,
                "config_params": config_params,
            }
        }
    else:
        data_logger = {"tensorboard": SummaryWriter(log_dir=save_path / "tensorboard")}

    ## Set the seed
    torch_gen = set_seed(cfg.random_seed)

    ## Set the device
    device = set_device()

    # Initialize a dummy c_mode
    models = None

    if (
        cfg.pretraining_checkpoint is None
        and cfg.resume_pretraining
        or (cfg.resume_pretraining and cfg.pretraining_checkpoint is not None)
        or (cfg.pretraining_eval_only and cfg.pretraining_checkpoint is not None)
    ):
        logger.info("")
        logger.info("[PRETRAINING]")

        ## Pretraining step
        # Create the pretraining object
        pretrain = Pretraining(
            server=cfg.server_model,
            client=cfg.client_model,
            head=cfg.classifier,
            device=device,
            data_folder=cfg.data_folder,
            checkpoint=cfg.pretraining_checkpoint,
            eval=cfg.pretraining_eval_only,
            data_logger=data_logger,
            dataloader_workers=cfg.pretraining_dataloader_workers,
        )
        # Load the pretraining dataset
        pretrain.set_data(
            dataset_info=cfg.pretrain_dts,  # w/ default datasets
            batch_size=cfg.pretraining_batch_size,
            generator=torch_gen,
            train_transforms=cfg.pretrain_data_tr,
            val_transforms=cfg.val_data_tr,
        )

        # Load the additional datasets
        # will instantiate the self.dataloaders_a1 (2)
        # and self.label_mapping_a1 (2) class variables
        if parse_def_conf(cfg.pretraining, "dts_a1") is not None:
            pretrain.set_data(
                dataset_info=cfg.pretraining.dts_a1,
                batch_size=cfg.pretraining_batch_size,
                generator=torch_gen,
                train_transforms=cfg.pretrain_data_tr,
                val_transforms=cfg.val_data_tr,
                is_additional=1,
            )

            if parse_def_conf(cfg.pretraining, "dts_a2") is not None:
                pretrain.set_data(
                    dataset_info=cfg.pretraining.dts_a2,  # w/ default datasets
                    batch_size=cfg.pretraining_batch_size,
                    generator=torch_gen,
                    train_transforms=cfg.pretrain_data_tr,
                    val_transforms=cfg.val_data_tr,
                    is_additional=2,
                )

        # now that the dataset are instantiated we can initialize the classifiers
        pretrain.init_classifiers()

        # Train the model
        models = pretrain(
            n_epochs=cfg.pretraining_epochs,
            optim=cfg.pretraining.optim,
            scheduler=parse_def_conf(cfg.pretraining, "scheduler"),
            log_every=cfg.pretraining_log_every,
            evaluate_every=cfg.pretraining_eval_every,
            loss=parse_def_conf(cfg.pretraining, "loss"),
            loss_weight=cfg.pretrain_loss_weight,
            kd_loss_weight=cfg.pretrain_kd_loss_weight,
            kd_loss=parse_def_conf(cfg.pretraining, "kd_loss"),
            save_path=save_path,
            early_stopping=cfg.pretraining_early_stopping,
            max_epoch_iters=cfg.pretraining_max_epoch_iters,
            module2train=cfg.pretraining_module2train,
        )

    if not cfg.skip_fed:
        logger.info("")
        logger.info("[FEDERATED TRAINING]")

        ## Federated Learning step
        # Define the model to load if pretraining is enabled
        # and not in eval only
        if (
            cfg.pretraining_checkpoint is None
            and cfg.federated_checkpoint is None
            and models is not None
        ):
            fed_client_model = models["model"]
            fed_head_model = models["head"]

        else:
            fed_client_model = cfg.client_model
            fed_head_model = cfg.classifier

        # Create the fed-trainer object
        fed_trainer = FedTrainer(
            s_model=cfg.server_model,
            c_model=fed_client_model,
            head=fed_head_model,
            n_classes=get_dts()[cfg.federated.dts.name]["n_classes"],
            criterion=cfg.federated.loss,
            optimizer=cfg.federated.optim,
            device=device,
            data_folder=cfg.data_folder,
            save_path=save_path,
            checkpoint=cfg.federated_checkpoint,
            dataloader_workers=parse_def_conf(cfg, "federated_dataloader_workers"),
            data_logger=data_logger,
        )

        # Load the federated dataset
        fed_trainer.set_data(
            dataset_info=cfg.federated.dts,
            batch_size=cfg.batch_size,
            partitioner=cfg.federated.partitioner,
            transforms=cfg.val_data_tr,
            seed=cfg.random_seed,
            generator=torch_gen,
        )

        ## Define the clients and server
        fed_trainer.set_client_and_server(
            client_type=cfg.federated.client_type,
            strategy=cfg.federated.strategy,
            num_rounds=cfg.num_rounds,
            log_every=cfg.save_fed_model_interval,
            eval_every=cfg.fed_eval_every_round,
            class_debias=cfg.fed_server_class_debias,
            class_debias_weight=cfg.fed_server_class_debias_weight,
            client_dp=parse_def_conf(cfg.federated, "client_dp"),
            server_dp=parse_def_conf(cfg.federated, "server_dp"),
        )

        ## Plot the label distribution
        if cfg.plot_label_distribution:
            fed_trainer.plot_partitions(
                cmap="prism", bar_plot_size=(50, 10), heatmap_size=(50, 80)
            )

        ## Run the simulation
        if not cfg.fed_eval_only:
            # Run the federated training
            logger.info("Running the federated training")
            fed_trainer(
                num_clients=cfg.num_clients,
                log_clients_print=cfg.log_clients_print,
            )
        else:
            # Evaluate the model on the server
            logger.info("Running only the evaluation of the server and client models")
            fed_trainer.eval_only()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
