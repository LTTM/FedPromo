"""
Module to handle the StanfordDogs dataset.
The module provide all the tools necessary to download
and initilize the dataset.
http://vision.stanford.edu/aditya86/ImageNetDogs/

Classes:
    StanfordDogs: StanfordDogs dataset wrapper.

Functions:
    None

Constants:
    SUPPORTED_SPLITS: List of supported splits.
    NUM_CLASSES: Number of classes in the dataset.
    NUM_SAMPLES: Number of samples for each split.

Exceptions:
    ValueError: Raised when the split is not supported.
    RuntimeError: Raised when the download fails.

Author: Matteo Caligiuri
"""

from typing import Union, Optional, Callable
from pathlib import Path
import json

from torchvision.datasets import ImageFolder

from modules.common.decorators import add_dts
from modules.common.logger import Logger, TqdmLogger
from .def_dataset import DefDataset


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


__all__ = ["StanfordDogs"]


# Constants
SUPPORTED_SPLITS = ["train", "val"]
NUM_CLASSES = 120
NUM_SAMPLES = {"train": 12000, "val": 8580}


@add_dts
class StanfordDogs(ImageFolder, DefDataset):
    """
    StanfordDogs dataset wrapper.
    This class downlaod the correct data automatically if the root folder doesn't exist.
    Extarct it and split it in the correct folders.
    Following the format expected by the ImageFolder class.
    It also process the classes to put them in the same format of the other datasets.

    Args:
        root (Union[str, Path]): The root folder where the dataset will be saved.
        split (str, optional): The split of the dataset.
            Defaults to "train".
        transform (Optional[Callable]): None.
        target_transform (Optional[Callable]): None.
        download (bool): If True, downloads the dataset from the internet and puts it
            in root directory. If dataset is already downloaded, it is not downloaded
            again. Default: False.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Optional[str] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        # Cast the root to a Path object
        self.root = Path(root) if isinstance(root, str) else root

        # Download the dataset (if needed and required)
        if download and (
            not (self.root / "train").exists()
            or not (self.root / "val").exists()
            or not (self.root / "classes.json").exists()
        ):
            raise NotImplementedError("Download is not implemented yet.")
        else:
            # Load the classes from the json file
            with open(self.root / "classes.json", "r", encoding="utf-8") as f:
                classes_dts = json.load(f)
            self.dts_classes = classes_dts
            del classes_dts

        # Check the splits
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Split {split} not supported. Supported splits: {SUPPORTED_SPLITS}"
            )

        # Initialize the dataset
        super().__init__(
            root=self.root / split,
            transform=transform,
            target_transform=target_transform,
        )

        # Define the image_paths attribute
        self.image_paths = [
            str(p).replace("\\", "/").replace(str(self.root).replace("\\", "/"), "")
            for p, _ in self.samples
        ]

        # define the embeddings var
        self.embeddings = None

        # Overwrite the classes and class_to_index attributes
        self.classes = [cls_name for cls_name in self.dts_classes]
        self.class_to_idx = self.dts_classes
        del self.dts_classes
