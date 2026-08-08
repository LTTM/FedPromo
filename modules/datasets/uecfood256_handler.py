"""
Module to handle the UECFOOD-256 dataset.
The module provide all the tools necessary to download
and initilize the UECFOOD-256 dataset.
http://foodcam.mobi/dataset256.html

Classes:
    UECFOOD256: UECFOOD-256 dataset wrapper.

Functions:
    None

Constants:
    SUPPORTED_SPLITS: List of supported splits.
    NUM_CLASSES: Number of classes in the dataset.
    NUM_SAMPLES: Number of samples for each split.
    DOWNLOAD_LINK: Link to download the dataset.
    SPLITS_DOWNLAOD_LINK: Link to download the splits.
    DATA_FOLDER: The folder where the data will be saved.
    SPLIT_FOLDER: The folder where the splits will be saved.

Exceptions:
    ValueError: Raised when the split is not supported.

Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Union, Optional, Callable, Dict
from pathlib import Path
import shutil
import json
from contextlib import redirect_stderr, redirect_stdout

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

from modules.common.decorators import add_dts
from modules.common.logger import Logger, TqdmLogger
from .def_dataset import DefDataset


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


__all__ = ["UECFOOD256"]


# Constants
SUPPORTED_SPLITS = ["train", "val"]
NUM_CLASSES = 256
NUM_SAMPLES = {"train": 25641, "val": 5754}
DOWNLOAD_LINK = "http://foodcam.mobi/dataset256.zip"
SPLITS_DOWNLAOD_LINK = "http://foodcam.mobi/uecfood_split.zip"
DATA_FOLDER = "UECFOOD256"
SPLIT_FOLDER = "uecfood_split"


@add_dts
class UECFOOD256(ImageFolder, DefDataset):
    """
    UECFOOD-256 dataset wrapper.
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
            self._process_splits()
        else:
            # Load the classes from the json file
            with open(self.root / "classes.json", "r", encoding="utf-8") as f:
                self.dts_classes = json.load(f)

        # Check the splits
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Split {split} not supported. Supported splits: {SUPPORTED_SPLITS}"
            )

        # save split, needed to load correct embeddings
        self.split = split

        # Initialize the dataset
        super().__init__(
            root=self.root / split,
            transform=transform,
            target_transform=target_transform,
        )

        # Define the image_paths attribute
        self.image_paths = [
            str(p)
            .replace("\\", "/")
            .replace(str(self.root).replace("\\", "/"), "")
            for p, _ in self.samples
        ]

        # define the embeddings var
        self.embeddings = None

        # Set the propper classes and class_to_idx attributes
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

        logger.info("--- DOWNLOADING THE UECFOOD-256 DATASET ---")

        ## Download the images
        if not (self.root / DATA_FOLDER).exists():
            logger.info("Downloading the dataset...")
            with redirect_stdout(tqdml), redirect_stderr(tqdml):
                download_and_extract_archive(
                    url=DOWNLOAD_LINK,
                    download_root=self.root,
                    extract_root=self.root,
                    remove_finished=True,
                )
            logger.info("Dataset downloaded.")
        else:
            logger.info("Dataset already downloaded.")

        ## Download the splits
        if not (self.root / SPLIT_FOLDER).exists():
            logger.info("Downloading the splits...")
            with redirect_stdout(tqdml), redirect_stderr(tqdml):
                download_and_extract_archive(
                    url=SPLITS_DOWNLAOD_LINK,
                    download_root=self.root,
                    extract_root=self.root / SPLIT_FOLDER,
                    remove_finished=True,
                )
            logger.info("Splits downloaded.")
        else:
            logger.info("Splits already downloaded.")

    def _process_splits(self) -> None:
        """
        Starting from the row downloaded data splits them in the correct folders.
        Following the format expected by the ImageFolder class.
        A folder for train and a folder for validation.

        Args:
            None

        Returns:
            None
        """

        logger.info("--- PROCESSING THE UECFOOD-256 DATASET ---")

        ## Define the image path
        image_path = self.root / DATA_FOLDER

        ## Define the train and val folders
        train_path = self.root / "train"
        val_path = self.root / "val"
        train_path.mkdir(exist_ok=True, parents=True)
        val_path.mkdir(exist_ok=True, parents=True)

        ## Load the split file
        logger.info("Loading the split files...")
        val_split_path = self.root / f"{SPLIT_FOLDER}/uecfood256_split/val0.txt"

        with open(val_split_path, "r", encoding="utf-8") as f:
            val_split = f.readlines()

        ## Extract the classes
        logger.info("Extracting the classes...")
        self.dts_classes = self._parse_classes(image_path)

        ## Create the folders
        for key in self.dts_classes:
            Path(train_path / str(key)).mkdir(exist_ok=True, parents=True)
            Path(val_path / str(key)).mkdir(exist_ok=True, parents=True)

        ## Move the images
        # Validation
        for line in tqdm(
            val_split, desc="Copying val images to the correct folder", file=tqdml
        ):
            img_name = line.strip()
            shutil.move(
                self.root / img_name,
                val_path / str(int(img_name.split("/")[-2]) - 1),
            )
        # Train
        training_images = list(image_path.glob("*/**/*.jpg"))
        for img in tqdm(
            training_images,
            desc="Copying train images to the correct folder",
            file=tqdml,
        ):
            shutil.move(
                img,
                train_path / str(int(img.parent.name) - 1),
            )

        ## Remove useless files and folders
        shutil.rmtree(image_path)
        shutil.rmtree(self.root / SPLIT_FOLDER)

    def _parse_classes(self, iamge_path: Path) -> Dict[int, str]:
        """
        Parse the classes from the downloaded data.

        Args:
            image_path (Path): The path to the images.

        Returns:
            Dict[int, str]: The classes in the correct format.
        """

        ## Extract the classes
        with open(iamge_path / "category.txt", "r", encoding="utf-8") as f:
            raw_cls = f.readlines()
        classes = {
            (int(cls.strip().split("\t")[0]) - 1): cls.strip().split("\t")[1]
            for cls in raw_cls[1:]
        }

        # Save the classes as a json file
        with open(self.root / "classes.json", "w", encoding="utf-8") as f:
            json.dump(classes, f)

        return classes

    def _remove_raw_data(self) -> None:
        """
        Remove the raw data.
        After the download and the split parsing, the original dataset data will be removed.

        Args:
            None

        Returns:
            None
        """

        logger.info("Removing the raw data...")
        shutil.rmtree(list(self.root.iterdir())[0])
        logger.info("Raw data removed.")
