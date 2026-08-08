"""
Module to handle the ImageNet-1k dataset.
The module provide all the tools necessary to download
and initilize the Imagenet-1k dataset.

Classes:
    ImageNet: ImageNet dataset wrapper.

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

from typing import Union, Any
from pathlib import Path

from torchvision.datasets import ImageNet as ImageNet_torch
from torchvision.datasets.utils import download_url

from modules.common.decorators import add_dts
from .def_dataset import DefDataset


__all__ = ["ImageNet"]


# Constants
SUPPORTED_SPLITS = ["train", "val"]
NUM_CLASSES = 1000
NUM_SAMPLES = {"train": 1281167, "val": 50000}


@add_dts
class ImageNet(ImageNet_torch, DefDataset):
    """
    ImageNet dataset wrapper.
    With respect to the standard torchvision implementation, this class
    downlaod the correct data automatically if the root folder doesnt exist.

    Args:
        root (Union[str, Path]): The root folder where the dataset will be saved.
        split (str, optional): The split of the dataset.
            Defaults to "train".
        version (str, optional): The version of the dataset.
            Defaults to "2012".
        target_type (Optional[Union[Callable, List[Callable]]], optional):
            The type of the target to return.
            Defaults to None.
        transform (Optional[Callable], optional): The transform to apply to the data.
            Defaults to None.
        target_transform (Optional[Callable], optional):
            The transform to apply to the target.
            Defaults to None.
    """

    def __init__(
        self, root: Union[str, Path], split: str = "train", **kwargs: Any
    ) -> None:
        # Check if the dataset has already been downloaded
        root = Path(root) if isinstance(root, str) else root
        if not root.exists():
            # Download the devkit file
            download_url(
                url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
                root=root,
            )
            # Download the train data
            if split == "train":
                download_url(
                    url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
                    root=root,
                )
            # Download the validation data
            elif split == "val":
                download_url(
                    url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
                    root=root,
                )
            else:
                raise ValueError(
                    f"Split {split} not supported, supported splits are {SUPPORTED_SPLITS}."
                )

        # save split, needed to load correct embeddings
        self.split = split

        # define the embeddings var
        self.embeddings = None

        # Initialize the dataset
        super().__init__(root=root, split=split, **kwargs)

        # Define the image_paths attribute
        self.image_paths = [
            str(p)
            .replace("\\", "/")
            .replace(str(self.root / Path(split)).replace("\\", "/"), "")
            for p, _ in self.samples
        ]
