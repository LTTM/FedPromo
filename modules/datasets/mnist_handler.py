"""
Module to handle the MNIST dataset.
The module provide all the tools necessary to download
and initilize the MNIST dataset.

Classes:
    MNIST: MNIST dataset wrapper.

Functions:
    None

Constants:
    SUPPORTED_SPLITS: List of supported splits.
    NUM_CLASSES: Number of classes in the dataset.
    NUM_SAMPLES: Number of samples for each split.

Exceptions:
    ValueError: Raised when the split is not supported.

Author: Matteo Caligiuri
"""

from typing import Union, Optional, Callable
from pathlib import Path

from torchvision.datasets import MNIST as MNIST_torch

from modules.common.decorators import add_dts
from .def_dataset import DefDataset


__all__ = ["MNIST"]


# Constants
SUPPORTED_SPLITS = ["train", "val"]
NUM_CLASSES = 10
NUM_SAMPLES = {"train": 60000, "val": 10000}


@add_dts
class MNIST(MNIST_torch, DefDataset):
    """
    MNIST dataset wrapper.
    With respect to the standard torchvision implementation, this class
    downlaod the correct data automatically if the root folder doesn't exist.

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
        # Check if the dataset has already been downloaded
        root = Path(root) if isinstance(root, str) else root

        # Check the splits
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Split {split} not supported. Supported splits: {SUPPORTED_SPLITS}"
            )

        # save split, needed to load correct embeddings
        self.split = split

        # Initialize the dataset
        train = True if split == "train" else False
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Override the default classes values removing the numeric representation
        self.classes = [cls.split(" - ")[1] for cls in self.classes]
