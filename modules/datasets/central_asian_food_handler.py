"""
Module to handle the CentralAsianFood dataset.
The module provide all the tools necessary to download
and initilize the CentralAsianFood dataset.
https://github.com/IS2AI/Central-Asian-Food-Dataset

Classes:
    CentralAsianFood: CentralAsianFood dataset wrapper.
    
Functions:
    None

Constants:
    SUPPORTED_SPLITS: List of supported splits.
    NUM_CLASSES: Number of classes in the dataset.
    NUM_SAMPLES: Number of samples for each split.
    DOWNLOAD_LINK: Link to download the dataset.

Exceptions:
    ValueError: Raised when the split is not supported.

Author: Matteo Caligiuri
"""

from typing import Union, Optional, Callable, Dict
from pathlib import Path
import shutil
import os
import json
from contextlib import redirect_stderr, redirect_stdout

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from modules.common.decorators import add_dts
from modules.common.logger import Logger, TqdmLogger
from .def_dataset import DefDataset


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


__all__ = ["CentralAsianFood"]


# Constants
SUPPORTED_SPLITS = ["train", "val", "test"]
NUM_CLASSES = 42
NUM_SAMPLES = {"train": 10969, "val": 2735, "test": 4200}
DOWNLOAD_LINK = (
    "https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/CAFD/CAFD.zip"
)


@add_dts
class CentralAsianFood(ImageFolder, DefDataset):
    """
    CentralAsianFood dataset wrapper.
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
        if download and not (self.root / split).exists():
            self._download()
        else:
            # Load the classes from the json file
            with open(self.root / "classes.json", "r", encoding="utf-8") as f:
                self.dts_classes = json.load(f)

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

        # Set the classes and class_to_idx attributes
        self.classes = list(self.dts_classes.values())
        self.class_to_idx = {v: k for k, v in self.dts_classes.items()}
        del self.dts_classes

    def _download(self) -> None:
        """
        Download the dataset from the internet.

        Args:
            None

        Returns:
            None
        """

        logger.info("--- DOWNLOADING THE CENTRAL ASIAN FOOD DATASET ---")
        logger.info("Downloading the dataset...")
        with redirect_stdout(tqdml), redirect_stderr(tqdml):
            download_and_extract_archive(
                url=DOWNLOAD_LINK,
                download_root=self.root,
                extract_root=self.root,
                remove_finished=True,
            )
        # Remove the __MACOSX folder and the .DS_Store file
        if (self.root / "__MACOSX").exists():
            shutil.rmtree(self.root / "__MACOSX")
        # find all the .DS_Store files and remove them
        ds_store = list(self.root.glob("**/.DS_Store"))
        for file in ds_store:
            os.remove(file)

        logger.info("Dataset downloaded.")

        # Parse the classes
        logger.info("Parsing the classes...")
        self.dts_classes = self._parse_classes()
        logger.info("Classes parsed.")

    def _parse_classes(
        self,
    ) -> Dict[int, str]:
        """
        Parse the classes from the downloaded data.

        Args:
            None

        Returns:
            Dict(int, str): The classes of the dataset.
        """

        ## ELoad all the classes
        # List all the subfolder of the train one
        raw_cls = (self.root / "train").iterdir()
        # Order the classses in alphabetical order
        sorted_cls = sorted(raw_cls)
        # Define the mapping
        classes = {int(i): cls.name for i, cls in enumerate(sorted_cls)}

        ## Save the classes as a json file
        with open(self.root / "classes.json", "w", encoding="utf-8") as f:
            json.dump(classes, f)

        return classes
