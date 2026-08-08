"""
Module to handle the DINOv2 model from torch.
It will contain all the needed code to handle the model and its outputs
in such a way that is compatible with the rest of the library.

Classes:
    DINOv2Classifier: Class to handle the DINOv2 model from torch.

Functions:
    instantiate_dinov2: Instantiate the DINOv2 model from torch.
    main: Main function of the module to run the simulation.

Constants:
    None

Escceptions:
    AssertionError: Raised when an unsupported number of layers is provided.
    ValueError: Raised when the number of classes is not defined in the kwargs.

Author: Matteo Caligiuri
"""

import sys
import io
import warnings
from pathlib import Path
from contextlib import redirect_stderr
from typing import Optional, Union, List, Tuple
from datetime import datetime as date

import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

# Add system path to import the modules when the script is executed as main
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.common.logger import Logger  # pylint: disable=wrong-import-position
from modules.common.general_setup import (  # pylint: disable=wrong-import-position
    set_device,
    set_seed,
)
from modules.common.decorators import get_dts  # pylint: disable=wrong-import-position
from modules.trainers.single_model_trainer import (  # pylint: disable=wrong-import-position
    Trainer,
)


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()


# Ignore tinm DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


__all__ = ["instantiate_dinov2"]


def instantiate_dinov2(
    pretrained_model: str,
    data_folder: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cpu"),  # pylint: disable=redefined-outer-name
    **kwargs,
) -> torch.nn.Module:
    """
    Instantiate the DINOv2 model from torch.

    Args:
        pretrained_model (str): The pretrained model.
        data_folder (Optional[Union[Path, str]], optional):
            The folder where the model is stored.
        device (torch.device, optional):
            The device where the model should be loaded.
            Defaults to torch.device("cpu").
        **kwargs: Additional keyword arguments.
            Must contain the number of classes (n_classes).

    Returns:
        torch.nn.Module: The DINOv2 model.
    """

    # Parse the kwargs
    if "n_layers" in kwargs:
        n_layers = kwargs["n_layers"]
    else:
        n_layers = 1

    # Load the model
    dino = DINOv2Classifier(
        pretrained_model=pretrained_model,
        n_layers=n_layers,  # pylint: disable=possibly-used-before-assignment
        data_folder=data_folder,
        device=device,
    )

    return dino


class DINOv2Classifier(torch.nn.Module):
    """
    Class to handle the DINOv2 model from torch.
    It also defines a custom head that can be used for image classification.

    Attributes:
        dino (torch.nn.Module): The DINOv2 model.
    """

    def __init__(
        self,
        pretrained_model: str,
        n_layers: int = 1,
        data_folder: Optional[Union[Path, str]] = None,
        device: torch.device = torch.device(
            "cpu"
        ),  # pylint: disable=redefined-outer-name
    ) -> None:
        """
        Initialize the DINOv2 model from torch.

        Args:
            pretrained_model (str): The pretrained model.
            n_classes (int): The number of classes.
            n_layers (int, optional):
                The number of layers to add to the custom head.
                Defaults to 1.
                Accepted values are 1 and 4.
            no_head (bool, optional):
            data_folder (Optional[Union[Path, str]], optional):
                The folder where the model is stored.
            device (torch.device, optional):
                The device where the model should be loaded.
                Defaults to torch.device("cpu").
        """

        super(DINOv2Classifier, self).__init__()

        # Check the number of layers
        if n_layers not in (1, 4):
            raise AssertionError(
                f"Unsupported number of layers: {n_layers}."
                + " Supported values are 1 and 4."
            )

        # Set download folder for torch.hub
        if data_folder is not None:
            data_folder = Path(data_folder)
            data_folder.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(data_folder)

        # Load the backbone from torch.hub
        self.backbone = self._load_backbone(pretrained_model)

        # Define the custom head
        self.feat_shape = self.backbone.norm.normalized_shape[0]

        # Move the model to the desired device
        self.backbone.to(device)
        self.device = device

        # Define class attributes
        self.layers = n_layers
        self.num_register_tokens = self.backbone.num_register_tokens
        self.patch_size = self.backbone.patch_size

    def _load_backbone(self, pretrained_model: str) -> torch.nn.Module:
        """
        Load the backbone of the model.

        Args:
            pretrained_model (str): The pretrained model.

        Returns:
            torch.nn.Module: The backbone of the model.
        """

        with warnings.catch_warnings(record=True) as w:
            # Catch all the UserWarning warnings raised by torch.hub.load
            warnings.simplefilter(action="always", category=UserWarning)

            logger.info("Loading DINOv2 model %s...", pretrained_model)
            # Load the model (if not already downloaded it will be downloaded)
            # and catch the output of the download process
            with redirect_stderr(io.StringIO()) as out:
                backbone = torch.hub.load(
                    repo_or_dir="facebookresearch/dinov2", model=pretrained_model
                )

            # Log the warnings and messages
            self._log_warn_and_msg(w, out.getvalue())
            logger.info("DINOv2 model %s succesfully loaded", pretrained_model)

        return backbone

    def forward(
        self, x: torch.Tensor, normalize: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            normalize (Optional[bool]): Normalize the output. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output of the model and the
                features extracted from the model.
        """

        if self.layers == 1:
            x = self.backbone.forward_features(
                x,
            )
            if normalize:
                cls_token = x["x_norm_clstoken"]
                patch_tokens = x["x_norm_patchtokens"]
            else:
                cls_token = x["x_prenorm"][:, 0]
                patch_tokens = x["x_prenorm"][:, self.num_register_tokens + 1 :]
            # feats = cls_token
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(
                x, n=4, return_class_token=True, norm=normalize
            )
            # feats = x[3][0]
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return linear_input

    @torch.no_grad()
    def get_class_scores(
        self,
        image: Optional[torch.Tensor] = None,
        image_feats: Optional[torch.Tensor] = None,
        normalize: Optional[bool] = True,
        reshape: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Get the class scores and features.

        Args:
            image (torch.Tensor): The image/s.
            image_feats (torch.Tensor): The image features (normalized).
            normalize (Optional[bool]): Normalize the image features. Defaults to True.
            reshape (Optional[bool]): Reshape the image features. Defaults to True.
            compute_pred (Optional[bool]): Compute the prediction. Defaults to False.
        Returns:
            torch.Tensor: The final features.
        """

        # Check if image or image_features are defined
        # and act accordingly to get the image features
        if image is not None:
            image_feats = self.forward(image, normalize=normalize)
        elif image is None and image_feats is None:
            raise ValueError("Either image or image_features must be defined.")

        # Reshape the image features if required
        if reshape:
            b, _, h, w = image.shape
            image_feats = (
                image_feats.reshape(b, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        # Return the features
        return image_feats

    def get_feats_shape(self) -> int:
        """
        Get the shape of the features.

        Returns:
            int: The shape of the features.
        """

        return self.feat_shape * 2

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

        # Convert the save path to a Path object (if provided)
        if save_path is not None and isinstance(save_path, str):
            save_path = Path(save_path)

        # Set the model to evaluation mode
        self.eval()

        # Initialize the prototypes and the counter
        proto = torch.zeros(n_classes, self.feat_shape * 2).to(self.device)
        counter = torch.zeros(n_classes).to(self.device)

        # Compute the prototypes
        # Cycle through the dataloder with a batrch size of 1
        for data in tqdm(
            dataloader.dataset,
            desc="Computing prototypes",
            total=len(dataloader.dataset),
        ):
            img, label = data[0].to(self.device), data[1]
            _, feats = self.forward(img.unsqueeze(0))
            proto[label] += feats[0].detach()
            counter[label] += 1

        proto /= counter.unsqueeze(1)

        # Set the model back to training mode
        self.train()

        # Save the prototypes if required
        if save_path is not None:
            torch.save(proto, save_path)

        return proto

    @staticmethod
    def _log_warn_and_msg(warn: List[warnings.WarningMessage], messages: str) -> None:
        """
        Log the warnings and messages.

        Args:
            warn (List[warnings.WarningMessage]): The warnings.
            messages (str): The messages.
        """

        # Log the messages
        # Parse the message
        msg = messages.split("\n")
        for m in msg:
            if "Downloading" in m and "main" in m:
                logger.info("Main model downloaded")
            if "Downloading" in m and ".pth" in m:
                ckpt_name = m.split(" to ")[-1].split("/")[-1]
                logger.info("Checkpoint %s downloaded", ckpt_name)
            if "cache" in m:
                logger.info("Model laoded from cache (%s)", m.split(" ")[-1])

        # Log the warnings
        xformer = {"available": [], "unavailable": []}
        for w in warn:
            msg = str(w.message)
            if w.category == UserWarning and "xFormers" in msg:
                if "not available" in msg:
                    xformer["unavailable"].append(msg.split("(")[1][:-1])
                else:
                    xformer["available"].append(msg.split("(")[1][:-1])
        if len(xformer["unavailable"]) > 0:
            logger.info(
                "xFormers is not available for: %s", ", ".join(xformer["unavailable"])
            )
        if len(xformer["available"]) > 0:
            logger.info(
                "xFormers is available for: %s", ", ".join(xformer["available"])
            )


###############################################################################


# Define the main function that will be used to test the module
# It will fine tune DINOv2 on the provided dataset
@hydra.main(config_path="../../conf", config_name="dinov2", version_base=None)
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
    model = instantiate(
        cfg.server_model,
        device=device,
        data_folder=cfg.data_folder,
        n_classes=get_dts()[cfg.pretraining.dts.name]["n_classes"],
    )
    head = instantiate(
        cfg.classifier,
        features=model.get_feats_shape(),
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

        # Set the scheduler
        if "scheduler" in cfg.keys():
            scheduler = cfg.pretraining.scheduler
        else:
            scheduler = None

        # Train the model
        trainer(
            n_epochs=cfg.epochs,
            scheduler=scheduler,
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
