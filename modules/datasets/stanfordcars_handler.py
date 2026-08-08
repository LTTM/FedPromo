"""
Module to handle the StanfordCars dataset.
The module provide all the tools necessary to download
and initilize the StanfordCars dataset.

Classes:
    StanfordCars: StanfordCars dataset wrapper.

Functions:
    None

Constants:
    SUPPORTED_SPLITS: List of supported splits.
    NUM_CLASSES: Number of classes in the dataset.

Exceptions:
    ValueError: Raised when the split is not supported.

Author: Matteo Caligiuri
"""

from typing import Union, Optional, Callable
from pathlib import Path

from torchvision.datasets import StanfordCars as StanfordCars_torch

from modules.common.decorators import add_dts
from .def_dataset import DefDataset


__all__ = ["StanfordCars"]


# Constants
SUPPORTED_SPLITS = ["train", "val"]
NUM_CLASSES = 196
NUM_SAMPLES = {"train": 8144, "val": 8041}


@add_dts
class StanfordCars(StanfordCars_torch, DefDataset):
    """
    StanfordCars dataset wrapper.
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
        # Cast the root to a Path object
        root = Path(root) if isinstance(root, str) else root

        # If the dts folder exists force download to False
        if root.exists():
            download = False

        # Check the splits
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Split {split} not supported. Supported splits: {SUPPORTED_SPLITS}"
            )

        # Convert the split name to the correct one val -> test
        if split == "val":
            split = "test"

        # save split, needed to load correct embeddings
        self.split = split

        # Initialize the dataset
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Define the image_paths attribute
        self.image_paths = [
            str(p)
            .replace("\\", "/")
            .replace(str(self._base_folder).replace("\\", "/"), "")
            for p, _ in self._samples
        ]

        # define the embeddings var
        self.embeddings = None

        # Override the default classes values removing the numeric representation
        self.classes = [cls.replace("_", " ") for cls in self.classes]

        # Define the targets attrivute
        self.targets = [target for _, target in self._samples]
