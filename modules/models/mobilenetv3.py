"""
Wrapper for the MobileNetV3 model.

Classes:
    None

Functions:
    initialize_mobilenetv3: Initialize the MobileNetV3 model.
    main: Main function to run the simulation.

Constants:
    ALLOWED_WEIGHTS: List of allowed weights.

Exceptions:
    ValueError: Raised when the provided weight is not allowed.

Author: Matteo Caligiuri
"""

from typing import Optional, Union
from pathlib import Path
from datetime import datetime as date
import sys
from contextlib import redirect_stderr, redirect_stdout

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models.mobilenetv3 import (
    MobileNetV3,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


# Add system path to import the modules when the script is executed as main
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.common.logger import (  # pylint: disable=wrong-import-position
    Logger,
    TqdmLogger,
)
from modules.common.general_setup import (  # pylint: disable=wrong-import-position
    set_device,
    set_seed,
)
from modules.common.decorators import (  # pylint: disable=wrong-import-position
    register_method,
    register_property,
    register_new_class_elements,
    get_dts,
)
from modules.trainers.single_model_trainer import (  # pylint: disable=wrong-import-position
    Trainer,
)
from modules.common.auxiliary_fn import (  # pylint: disable=wrong-import-position
    parse_def_conf,
)


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


__all__ = ["initialize_mobilenetv3"]


# Set the target class for the decorators
_target_class = MobileNetV3  # pylint: disable=invalid-name


# Constants
ALLOWED_WEIGHTS = ["IMAGENET1K_V1", "IMAGENET1K_V2", "DEFAULT"]


def initialize_mobilenetv3(
    size: str,
    weight: Optional[str] = None,
    out_feats_size: Optional[int] = None,
    n_layers_translator: Optional[int] = 1,
) -> MobileNetV3:
    """
    Initialize the MobileNetV3 model.

    Args:
        size (str): The size of the model.
        weight (str): The weight of the model.
        num_classes (Optional[int]): The number of classes.
        out_feats_size (Optional[int]): The size of the output features.
        n_layers_translator (Optional[int]): The number of layers for the translator.
    Returns:
        MobileNetV3: The MobileNetV3 model.
    """

    # Check if the provided weights are allowed
    weight = weight.upper()
    if weight not in ALLOWED_WEIGHTS:
        raise ValueError(f"Unknown weight: {weight}")

    if size == "small":
        with redirect_stdout(tqdml), redirect_stderr(tqdml):
            model = mobilenet_v3_small(
                weights=getattr(MobileNet_V3_Small_Weights, weight)
            )
    elif size == "large":
        with redirect_stdout(tqdml), redirect_stderr(tqdml):
            model = mobilenet_v3_large(
                weights=getattr(MobileNet_V3_Large_Weights, weight)
            )
    else:
        raise ValueError(f"Unknown size: {size}")

    # Add the new methods and properties to the model
    register_new_class_elements(model)

    # Define the feats_mapping property
    if out_feats_size is not None:
        model.set_feats_mapping(out_feats_size, n_layers_translator)

    # Remove the classifier
    model.classifier = nn.Identity()

    return model


@register_property
def feats_mapping() -> None:
    """
    Add the feats_mapping property to the MobileNetV3 class.
    """

    return None


@register_method
def set_feats_mapping(
    self, out_feats_size: Optional[int] = None, n_layers: Optional[int] = 1
) -> None:
    """
    Set the feats_mapping property.

    Args:
        out_feats_size (Optional[int]): The size of the output features.
        n_layers (Optional[int]): The number of layers.

    Returns:
        None
    """

    if out_feats_size is None:
        self.feats_mapping = None
    else:
        if n_layers == 1:
            self.feats_mapping = nn.Linear(576, out_feats_size, bias=False)
            # self.feats_mapping = nn.Sequential(
            #     nn.Linear(576, out_feats_size, bias=False),
            #     nn.LayerNorm(out_feats_size),
            # )
        elif n_layers >= 2:
            # Define the size of the intermediate layers based on the output size
            intermediate_size = out_feats_size * 2

            # Define the model
            layers = [nn.Linear(576, intermediate_size), nn.ReLU()]
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(intermediate_size, intermediate_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(intermediate_size, out_feats_size))
            self.feats_mapping = nn.Sequential(*layers)
        else:
            raise ValueError(f"Invalid number of layers: {n_layers}")


@register_method
def get_feats_shape(self) -> int:
    """
    Get the features shape.

    Args:
        None

    Returns:
        int: The features shape.
    """

    return self.features[-1].out_channels


@register_method
def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward method for the MobileNetV3 model.
    (Overrided from the original one to return also the features)

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The logits and the features.
    """

    x = self.features(x)

    x = self.avgpool(x)
    f = torch.flatten(x, 1)

    if self.feats_mapping is not None:
        f = self.feats_mapping(f)

    return f


@register_method
def get_device(self) -> torch.device:
    """
    Get the device of the model.

    Args:
        None

    Returns:
        torch.device: The device of the model.
    """

    return next(self.parameters()).device


@register_method
@torch.no_grad()
def create_prototypes(
    self,
    dataloader: torch.utils.data.DataLoader,
    n_classes: int,
    save_path: Optional[Union[Path, str]] = None,
) -> torch.Tensor:
    """
    Create the prototypes for the classes.

    Args:
        dataloader (torch.utils.data.Dataloader): The data loader.
        n_classes (int): The number of classes.
        save_path (Optional[Union[Path, str]], optional): The path where to save the prototypes.
            Defaults to None.

    Returns:
        torch.Tensor: The prototypes.
    """

    # Set the model in evaluation mode
    self.eval()

    # Convert the save path to a Path object (if provided)
    if save_path is not None and isinstance(save_path, str):
        save_path = Path(save_path)

    # Initialize the prototypes and the counter
    proto = torch.zeros(n_classes, self.feats_mapping.out_features).to(
        self.get_device()
    )
    counter = torch.zeros(n_classes).to(self.get_device())

    # Compute the prototypes
    # Cycle through the dataloder with a batrch size of 1
    index = 0
    for data in tqdm(
        dataloader.dataset, desc="Computing prototypes", total=len(dataloader.dataset)
    ):
        img, label = data[0].to(self.get_device()), data[1]
        feats = self.forward(img.unsqueeze(0))["feats"]
        index += 1
        proto[label] += feats[0].detach()
        counter[label] += 1

    proto /= counter.unsqueeze(1)

    # Set back the model in training mode
    self.train()

    # Save the prototypes if required
    if save_path is not None:
        torch.save(proto, save_path)

    return proto


###############################################################################


@hydra.main(config_path="../../conf", config_name="mobilenetv3", version_base=None)
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
        data_logger = {"mlflow": run_name}
    else:
        data_logger = {"tensorboard": SummaryWriter(log_dir=save_path / "tensorboard")}

    ## Set the seed
    torch_gen = set_seed(cfg.random_seed)

    ## Set the device
    device = set_device()

    ## Define the model
    # Get the features shape from the checkpoint if it is provided
    # otherwise get it from the configuration file
    if cfg.checkpoint is not None:
        f_size = _get_feats_shape(cfg.checkpoint["path"])
    elif cfg.feats_size is not None:
        f_size = cfg.feats_size
    else:
        f_size = 2_048

    # Initialize the model
    model = instantiate(
        cfg.client_model,
        out_feats_size=f_size,
    )
    head = instantiate(
        cfg.classifier,
        features=f_size,
        num_classes=get_dts()[cfg.pretraining.dts.name]["n_classes"],
    )

    ## Training and evaluation step
    trainer = Trainer(
        model=model,
        head=head,
        n_classes=get_dts()[cfg.pretraining.dts.name]["n_classes"],
        device=device,
        loss=(
            cfg.pretraining.loss
            if not cfg.proto_train
            else cfg.pretraining.prototype_distance
        ),
        optimizer=cfg.pretraining.optim,
        module2train=cfg.module2train,
        proto_train=cfg.proto_train,
        data_folder=cfg.data_folder,
        save_folder=save_path,
        checkpoint=cfg.checkpoint,
        eval=cfg.eval_only,
        data_logger=data_logger,
        log_every=cfg.log_every,
        evaluate_every=cfg.eval_every,
        early_stopping=cfg.early_stopping,
    )

    # Load the pretraining dataset
    if (
        cfg.checkpoint is None
        or (cfg.resume_train and (cfg.checkpoint is not None))
        or (cfg.eval_only and (cfg.checkpoint is not None))
    ):
        trainer.set_data(
            dataset_info=cfg.pretraining.dts,
            batch_size=cfg.batch_size,
            generator=torch_gen,
            transforms={
                "train": instantiate(cfg.train_data_tr),
                "val": instantiate(cfg.val_data_tr),
            },
        )

        # Train the model
        trainer(
            n_epochs=cfg.epochs,
            scheduler=parse_def_conf(cfg.pretraining, "scheduler"),
        )


def _get_feats_shape(ckpt_path: Union[str, Path]) -> int:
    """
    Get the features shape from the checkpoint.

    Args:
        ckpt_path (Union[str, Path]): The path to the checkpoint.

    Returns:
        int: The features shape.
    """

    # Load the model
    model = torch.load(ckpt_path, weights_only=False)["model"]

    # Get the features shape
    try:
        return model["feats_mapping.weight"].shape[0]
    except KeyError:
        return model["feats_mapping.0.weight"].shape[0]


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
