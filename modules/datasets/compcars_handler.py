"""
Module to handle the ComprehensiveCars dataset.
The module provide all the tools necessary to download
and initilize the ComprehensiveCars dataset.
http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html

If the download fails with the following error:
"Too many users have viewed or downloaded this file recently."
You can fix the problem by following these stesps:
1. Install on your browser the "Get cookies.txt LOCALLY" browser extension.
   Chrome:
   https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc?pli=1
   Mozzilla:
   https://addons.mozilla.org/en-US/firefox/addon/get-cookies-txt-locally/
2. Open the dataset link in your browser.
3. Click on the extension icon and download the cookies.txt file.
4. Place the cookies.txt file in "~/.cache/gdown/".
5. Run the code again.

Classes:
    CompCars: ComprehensiveCars dataset wrapper.

Functions:
    None

Constants:
    SUPPORTED_SPLITS: List of supported splits.
    NUM_CLASSES: Number of classes in the dataset.
    NUM_SAMPLES: Number of samples for each split.
    NUM_CLASSES_ALT: Number of classes in the dataset considering the model year.
    GDRIVE_LINK: The link to download the dataset from Google Drive.
    ZIP_PW: The password to unzip the dataset.
    SPLIT_FOLDER_NAME: The name of the folder containing the split files.
    SYSTEMS: A dictionary mapping the platform output to the supported systems.
    FILE_STRUCTURES: A dictionary containing the structure of the downloaded files.

Exceptions:
    ValueError: Raised when the split is not supported.
    RuntimeError: Raised when the download fails.

Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Union, Optional, Callable, Dict
from pathlib import Path
import os
import shutil
import platform
import subprocess
import json
from contextlib import redirect_stderr, redirect_stdout

from torchvision.datasets import ImageFolder
import gdown
from tqdm import tqdm
import scipy.io

from modules.common.decorators import add_dts, add_dts_alt
from modules.common.logger import Logger, TqdmLogger
from .def_dataset import DefDataset


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


__all__ = ["CompCars", "CompCarsYear"]


# Constants
SUPPORTED_SPLITS = ["train", "val"]
NUM_CLASSES = 431
NUM_SAMPLES = {"train": 16016, "val": 14939}
NUM_CLASSES_ALT = 1342
G_DRIVE_LINK = (
    "https://drive.google.com/drive/folders/18EunmjOJsbE5Lh9zA0cZ4wKV6Um46dkg"
)
ZIP_PW = "d89551fd190e38"
SPLIT_FOLDER_NAME = "train_test_split"
SYSTEMS = {"Unix": ["Linux", "Darwin"], "Windows": ["Windows"]}
FILE_STRUCTURES = {
    "std": ["data.zip"] + [f"data.z{i:02d}" for i in range(1, 23)],
    "sv": ["sv_data.zip"] + [f"sv_data.z{i:02d}" for i in range(1, 4)],
}


@add_dts
class CompCars(ImageFolder, DefDataset):
    """
    CompCars dataset wrapper.
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
        use_year_as_class (bool, optional): If True, the dataset classes will consider
            also the model year. Default: False.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Optional[str] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        use_year_as_class: Optional[bool] = False,
    ) -> None:
        # Cast the root to a Path object
        self.root = Path(root) if isinstance(root, str) else root

        # Download the dataset (if needed and required)
        if download and (
            not (self.root / "train").exists()
            or not (self.root / "val").exists()
            or not (self.root / "classes.json").exists()
        ):
            if not (self.root / "train").exists() or not (self.root / "val").exists():
                self._download()
            self._process_splits(model_year=use_year_as_class)
            self._remove_raw_data()
        else:
            # Load the classes from the json file
            with open(self.root / "classes.json", "r", encoding="utf-8") as f:
                classes_dts = json.load(f)
            self.dts_classes = classes_dts["verbose_classes"]
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
            str(p)
            .replace("\\", "/")
            .replace(str(self.root).replace("\\", "/"), "")
            for p, _ in self.samples
        ]

        # define the embeddings var
        self.embeddings = None

        # Overwrite the classes and class_to_index attributes
        self.classes = [v for v in self.dts_classes.values()]
        self.class_to_idx = {v: k for k, v in self.dts_classes.items()}
        del self.dts_classes

    def _download(self) -> None:
        """
        Download the dataset from the internet.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: Raised when the download fails.
        """

        logger.info("--- DOWNLOADING THE COMP-CARS DATASET ---")

        ## Download the images
        if (
            all((self.root / file).exists() for file in FILE_STRUCTURES["std"])
            or Path(self.root / "data").exists()
        ):
            logger.info("Dataset already downloaded.")
        else:
            logger.info("Downloading the dataset...")
            with redirect_stdout(tqdml), redirect_stderr(tqdml):
                gdown.download_folder(
                    url=G_DRIVE_LINK,
                    output=str(self.root),
                    quiet=False,
                    remaining_ok=True,
                    use_cookies=True,
                )
            logger.info("Dataset downloaded.")

        ## Unzip the images
        if Path(self.root / "data").exists():
            logger.info("Dataset already unzipped.")
        else:
            self._unzip_protected_zip(self.root / FILE_STRUCTURES["std"][0], ZIP_PW)

            # Remove all the unnecessary files
            logger.info("Removing unnecessary files...")
            for file in FILE_STRUCTURES["std"]:
                if (self.root / file).exists():
                    os.remove(self.root / file)
            for file in FILE_STRUCTURES["sv"]:
                if (self.root / file).exists():
                    os.remove(self.root / file)
            os.remove(self.root / "README.txt")
            logger.info("Unnecessary files removed.")

    def _process_splits(
        self,
        model_year: Optional[bool] = False,
        use_verification_split: Optional[bool] = False,
    ) -> None:
        """
        Starting from the row downloaded data splits them in the correct folders.
        Following the format expected by the ImageFolder class.
        A folder for train and a folder for validation.

        Args:
            model_year (bool, optional): If True, the dataset classes will
                consider also the model year.
            use_verification_split (bool, optional): If True, the dataset will
                use the verification split together withj the standars train split

        Returns:
            None
        """

        logger.info("--- PROCESSING THE COMP-CARS DATASET ---")

        ## Define the image path
        iamge_path = self.root / "data/image"

        ## Define the train and val folders
        train_path = self.root / "train"
        val_path = self.root / "val"
        train_path.mkdir(exist_ok=True, parents=True)
        val_path.mkdir(exist_ok=True, parents=True)

        ## Load the split file
        logger.info("Loading the split files...")
        train_split_path = self.root / "data/train_test_split/classification/train.txt"
        val_split_path = self.root / "data/train_test_split/classification/test.txt"
        verification_split = (
            self.root / "data/train_test_split/verification/verification_train.txt"
        )

        with open(train_split_path, "r", encoding="utf-8") as f:
            train_split = f.readlines()
        with open(val_split_path, "r", encoding="utf-8") as f:
            val_split = f.readlines()

        if use_verification_split:
            with open(verification_split, "r", encoding="utf-8") as f:
                verification_split = f.readlines()
            train_split += verification_split

        ## If model year is used, drop all the unknown year folder
        if model_year:
            train_split = [x for x in train_split if x.split("/")[2] != "unknown"]
            val_split = [x for x in val_split if x.split("/")[2] != "unknown"]

        ## Extract the classes
        logger.info("Extracting the classes...")
        drop_index = -1 if model_year else -2
        classes_dts = self._parse_classes(train_split + val_split, drop_index)
        classes = classes_dts["classes"]
        self.dts_classes = classes_dts["verbose_classes"]
        del classes_dts

        ## Create the folders
        for key in classes.values():
            Path(train_path / key).mkdir(exist_ok=True, parents=True)
            Path(val_path / key).mkdir(exist_ok=True, parents=True)

        ## Copy the images
        for line in tqdm(
            train_split, desc="Copying train images to the correct folder", file=tqdml
        ):
            img_name = line.strip()
            shutil.copy2(
                iamge_path / img_name,
                train_path / classes["/".join(img_name.split("/")[:drop_index])],
            )
        for line in tqdm(
            val_split, desc="Copying val images to the correct folder", file=tqdml
        ):
            img_name = line.strip()
            shutil.copy2(
                iamge_path / img_name,
                val_path / classes["/".join(img_name.split("/")[:drop_index])],
            )

        ## Remove all the empty folders
        logger.info("Removing empty folders...")
        for folder in train_path.glob("**/*"):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        for folder in val_path.glob("**/*"):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        logger.info("Empty folders removed.")

    def _parse_classes(
        self, split_file: Path, drop_index: int
    ) -> Dict[str, Dict[Union[str, int], str]]:
        """
        Parse the classes from the downloaded data.

        Args:
            split_file (Path): The path to the split file.
            drop_index (int): The index to drop from the class.

        Returns:
            Dict(str, Dict(Union[str, int], str)): The classes and the verbose classes.
        """

        ## Extract the classes
        # Drop the model year if needed and ignore the image name
        raw_cls = set(["/".join(x.strip().split("/")[:drop_index]) for x in split_file])
        # Define the lambda function for sorting
        sorted_cls = sorted(raw_cls, key=class_sort_key)
        # Define the mapping
        final_classes = {int(i): cls for i, cls in enumerate(sorted_cls)}
        classes = {cls: f"{i:03d}" for i, cls in enumerate(sorted_cls)}

        ## Get the verbose classes
        # Load the .mat file
        mat_path = self.root / "data/misc/make_model_name.mat"
        mat = scipy.io.loadmat(str(mat_path))
        # Extract the makers and models
        makers = [x.item().item() for x in mat["make_names"]]
        models = [
            x.item().item() if x.item().size > 0 else None for x in mat["model_names"]
        ]
        del mat
        # Define the mapping
        verbose_classes = {
            classes[cls]: (
                f"{makers[int(cls.split('/')[0]) - 1]} {models[int(cls.split('/')[1]) - 1]}"
                if drop_index == -2
                else f"{makers[int(cls.split('/')[0]) - 1]}"
                + f"{models[int(cls.split('/')[1]) - 1]}"
                + f"({cls.split('/')[2]})"
            )
            for cls in sorted_cls
        }

        # Save the classes as a json file
        with open(self.root / "classes.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "classes": classes,
                    "verbose_classes": verbose_classes,
                },
                f,
            )

        return {
            "final_classes": final_classes,
            "classes": classes,
            "verbose_classes": verbose_classes,
        }

    @staticmethod
    def _unzip_protected_zip(zip_file, zip_pw):
        # Check the platform
        system = platform.system()

        # Fuse the multi-part zip and unzip it
        if system in SYSTEMS["Unix"]:
            # Define the temporary file to use as combined zip
            combined_file = zip_file.parent / "combined.zip"

            # Fuse the multi-part zip
            logger.info("Fusing the multi-part zip...")
            with redirect_stdout(tqdml), redirect_stderr(tqdml):
                subprocess.run(
                    ["zip", "-F", zip_file, "--out", combined_file],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            logger.info("Multi-part zip fused.")

            # Unzip the combined zip
            logger.info(
                "Unzipping the combined zip (it could take a couple of minutes)..."
            )
            with redirect_stdout(tqdml), redirect_stderr(tqdml):
                subprocess.run(
                    ["unzip", "-P", zip_pw, combined_file.name],
                    check=True,
                    cwd=zip_file.parent,
                    stdout=subprocess.DEVNULL,
                )
            logger.info("Combined zip unzipped.")

            # Remove the temporary file
            os.remove(combined_file)
        elif system in SYSTEMS["Windows"]:
            # Unzip the zip
            logger.info("Unzipping the zip (it could take a couple of minutes)...")
            with redirect_stdout(tqdml), redirect_stderr(tqdml):
                subprocess.run(
                    [
                        str(Path("extras/7z2408-extra/7za.exe")),
                        "x",
                        zip_file,
                        f"-p{zip_pw}",
                    ],
                    check=True,
                    cwd=zip_file.parent,
                    stdout=subprocess.DEVNULL,
                )
            logger.info("Zip unzipped.")
        else:
            raise RuntimeError(f"System {system} not supported.")

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
        shutil.rmtree(self.root / "data")
        logger.info("Raw data removed.")


@add_dts_alt
class CompCarsYear(CompCars):
    """
    CompCars dataset wrapper.
    This class downlaod the correct data automatically if the root folder doesn't exist.
    Extarct it and split it in the correct folders.
    Following the format expected by the ImageFolder class.
    It also process the classes to put them in the same format of the other datasets.
    This class is a subclass of the CompCars class. It uses the same methods and attributes.
    But the model year is considered as a different class.

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

        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
            use_year_as_class=True,
        )


def class_sort_key(item: str) -> tuple:
    """
    Sort key for the classes.
    The key is based on the numeric representation of the class.
    maker/model or maker/model/year

    Args:
        item (str): The class to sort.

    Returns:
        tuple: The numeric representation of the class.
    """

    return tuple(int(part) for part in item.split("/"))
