"""
Module to handle the INaturalist dataset.
The module provide all the tools necessary to procces the iNaturalist dataset
and format it in standard format for the PyTorch framework.

Classes:
    INaturalist: The class to handle the INaturalist dataset.

Functions:
    _split_and_version_handler: Handle the split based on the version.
    _dw_split_file: Download the split file based on the version.
    _filter_dts: Filter the dataset based on the split.
    _classes_parser: Parse the classes file of the dataset.

Constants:
    ACCEPTED_SPLIT: List of accepted splits.
    SPLIT_MAP: Dictionary to map the split based on the version.
    VERSIONS_WITH_SPLIT: List of versions that require a split.
    SPLITS_URL: Dictionary of the split urls based on the version.
    CLASSES_URL: Dictionary of the classes urls based on the version.
    NUM_CLASSES: Number of classes in the dataset.
    NUM_SAMPLES: Number of samples for each split.

Exceptions:
    ValueError: If the version is not supported.
    ValueError: If the split is not supported.
    KeyError: If the split is not in the SPLITS_URL dictionary.

Author: Matteo Caligiuri
"""

import os
import shutil
from typing import Callable, Optional, Union, List, Tuple, Dict, Any
from pathlib import Path
import json

from PIL import Image
from omegaconf import ListConfig
from torchvision.datasets import INaturalist as INaturalist_torch
from torchvision.datasets.utils import download_and_extract_archive

from modules.common.decorators import add_dts
from .def_dataset import DefDataset


__all__ = ["INaturalist"]


# Constants
ACCEPTED_SPLIT = ["train", "train_mini", "val"]
SPLIT_MAP = {"train": "train", "train_mini": "train", "val": "val"}
VERSIONS_WITH_SPLIT = ["2021"]
SPLITS_URL = {
    "2017": {
        "train_val": "https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val2017.zip"
    },
    "2018": {
        "train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz",
        "val": "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz",
    },
    "2019": {
        "train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train2019.json.tar.gz",
        "val": "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/val2019.json.tar.gz",
    },
}
CLASSES_URL = {
    "2017": "",
    "2018": "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/categories.json.tar.gz",
    "2019": "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/categories.json.tar.gz",
    "2021": "",
}
NUM_CLASSES = 8142
NUM_SAMPLES = {"train": 437513, "val": 24426}


@add_dts
class INaturalist(INaturalist_torch, DefDataset):
    """
    INaturalist dataset wrapper.
    With respect to the standard torchvision implementation, this class
    downlaod the correct data from the server based on the version and split
    specified by the user. Than for the version that doesn't have a split in the version
    (e.g. 2021) the split is handled by the class itself. To do so it will downlaod the additional
    split file and filter the dataset based on them. Finally, the class also download the classes
    unobfuscated file and parse it to get the classes, superclasses and full classes names.
    As a bonus feature, if the download flag is set to True, but the dataset is already downloaded,
    the class will not download it again.

    Args:
        root (Union[str, Path]): The root directory to save the dataset.
        version (str): The version of the dataset. Default: "2018".
        target_type (Union[List[str], str]): Type of target to return. Default: "full".
        transform (Optional[Callable]): A function/transform that takes in an PIL image
            and returns a transformed version. Default: None.
        target_transform (Optional[Callable]): A function/transform that takes in the
            target and transforms it. Default: None.
        split (Optional[str]): The split to download. Default: "train".
        download (bool): If True, downloads the dataset from the internet and puts it
            in root directory. If dataset is already downloaded, it is not downloaded
            again. Default: False.
    """

    def __init__(
        self,
        root: Union[str, Path],
        version: str = "2018",
        target_type: Union[List[str], str] = "full",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split: Optional[str] = "train",
        download: bool = False,
    ) -> None:
        # Check if the dataset has already been downloaded
        # and set the download flag accordingly
        root = Path(root) if isinstance(root, str) else root
        download = not root.exists()

        # Check that the version and split of the dataset is supported
        # Then properly parse them
        info = _split_and_version_handler(split, root, version)
        year = version
        self.split = info["split"]
        version = info["version"]

        # Parse the target_type parameter
        if version in SPLITS_URL:
            categories = target_type
            target_type = "full"

        super().__init__(
            root=root,
            version=version,
            target_type=target_type,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Define the classes properties of the dataset
        classes = _classes_parser(root, year)
        self.full_classes = classes["full_classes"]
        self.classes = classes["fine_classes"]
        self.superclasses = classes["superclasses"]

        # Define the categories map
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Define the targets attribute
        self.targets = [target for target, _ in self.index]

        # Filter the dataset based on the split if the version requires it
        if version in SPLITS_URL:
            self.index = _filter_dts(
                self.split, root, version, self.superclasses, categories
            )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Override of the standard __getitem__ method.
        The only difference is that if the image is not in RGB format
        it will be converted to RGB.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        # Convert the image to RGB if it is in CMYK
        if img.mode == "CMYK":
            img = img.convert("RGB")

        target: Any = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def _split_and_version_handler(split: str, root: Path, version: str) -> Dict[str, str]:
    """
    Handle the split based on the version.

    Args:
        split (str): The split to download.
        root (Path): The root directory to save the split file.
        version (str): The version of the dataset.

    Returns:
        Dict[str, str]: The split and version information.

    Raises:
        ValueError: If the version is not supported.
        ValueError: If the split is not supported.
    """

    # Check if the split is supported
    if split not in ACCEPTED_SPLIT:
        raise ValueError(
            f"Split {split} not supported. Accepted values are {ACCEPTED_SPLIT}."
        )

    # Handle the split based on the version
    if version in SPLITS_URL:
        # Download train and val splits
        _dw_split_file(split, root, version)

        return {"split": split, "version": version}
    elif version in VERSIONS_WITH_SPLIT:
        return {"split": SPLIT_MAP[split], "version": f"{version}_{split}"}
    else:
        raise ValueError(
            f"Version {version} not supported. Accepted values are"
            + f"{VERSIONS_WITH_SPLIT + list(SPLITS_URL.keys())}."
        )


def _dw_split_file(split: str, root: Path, version: str) -> None:
    """
    Download the split file based on the version.

    Args:
        split (str): The split to download.
        root (Path): The root directory to save the split file.
        version (str): The version of the dataset.

    Returns:
        None

    Raises:
        KeyError: If the split is not in the SPLITS_URL dictionary.

    """

    try:
        # Download the train or val split file based on the split
        # For 2018, 2019 the train and val splits are separate
        if not (root / f"{split}{version}.json").exists():
            download_and_extract_archive(
                url=SPLITS_URL[version][split],
                download_root=root,
                extract_root=root,
                remove_finished=True,
            )
    except KeyError:
        # Download the train or val split file based on the split
        # For 2017 the train and val splits are together
        if not (root / f"{split}{version}.json").exists():
            download_and_extract_archive(
                url=SPLITS_URL[version]["train_val"],
                download_root=root,
                extract_root=root,
                remove_finished=True,
            )
            # Copy the train.yaml or val.yaml files to the root (based on the split)
            # and remove the train_val folder
            # if not already done by the download_and_extract_archive function
            try:
                if split == "train":
                    shutil.copy(
                        root / f"train_val{version}/train{version}.json",
                        root / f"train{version}.json",
                    )
                elif split == "val":
                    shutil.copy(
                        root / f"train_val{version}/val{version}.json",
                        root / f"val{version}.json",
                    )
                shutil.rmtree(root / f"train_val{version}")
            except FileNotFoundError:
                pass


def _filter_dts(
    split: str,
    root: Path,
    version: str,
    superclasses: List[str],
    categories: Union[List[str], str] = "full",
) -> List[Tuple[int, str]]:
    """
    Filter the dataset based on the split.

    Args:
        split (str): The split to filter.
        root (Path): The root directory to save the split file.
        version (str): The version of the dataset.
        superclasses (List[str]): The superclasses of the dataset.
        categories (Union[List[str], str]): The categories to filter. Default: "full".

    Returns:
        List[Tuple[int, str]]: The filtered dataset.
    """

    # Load the split json file
    with open(file=root / f"{split}{version}.json", encoding="utf-8") as f:
        data = json.load(f)
        data.pop("info")
        data.pop("licenses")
        data.pop("annotations")
        data.pop("categories")

    # Parse the category param
    if isinstance(categories, str):
        categories = [categories.capitalize()]
    elif isinstance(categories, list):
        categories = [cat.capitalize() for cat in categories]
    elif isinstance(categories, ListConfig):
        categories = list(categories)
        categories = [cat.capitalize() for cat in categories]

    # Rewrite the index property of the dataset object
    if categories[0] == "Full":
        indexes = [
            (int(img["file_name"].split("/")[2]), img["file_name"].split("/")[-1])
            for img in data["images"]
        ]
    else:
        # Filter the dataset based on the categories
        indexes = []
        for img in data["images"]:
            if superclasses[int(int(img["file_name"].split("/")[2]))] in categories:
                indexes.append(
                    (
                        int(img["file_name"].split("/")[2]),
                        img["file_name"].split("/")[-1],
                    )
                )

    return indexes


def _classes_parser(root: Path, year: str) -> Dict[str, List[str]]:
    """
    Parse the classes file of the dataset.

    Args:
        root (Path): The root directory of the dataset.
        year (str): The year of the dataset.

    Returns:
        Dict{str, List[str]}: The classes of the dataset.
    """

    # Download the unobfuscated classes file
    if not Path(root / "categories.json").exists():
        try:
            download_and_extract_archive(
                url=CLASSES_URL[year],
                download_root=root,
                extract_root=root,
                remove_finished=True,
            )
        except RecursionError as e:
            raise ValueError(f"Classes file for version {year} not found.") from e

    # Load the classes file
    with open(file=root / "categories.json", encoding="utf-8") as f:
        data = json.load(f)

    full_classes = [
        f"{elm['supercategory']}/{elm['family']}/{elm['name']}" for elm in data
    ]
    fine_classes = [elm["name"] for elm in data]
    superclasses = [elm["supercategory"] for elm in data]

    return {
        "full_classes": full_classes,
        "fine_classes": fine_classes,
        "superclasses": superclasses,
    }
