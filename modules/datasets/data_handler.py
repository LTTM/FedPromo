"""
Module to handle the datasets.

Classes:
    _Transform: Apply the transformations to the data.
        Required to handle the dict returned by the huggingface datasets library.
    _FedDataLoader: Load the federated datasets.

Functions:
    load_dts: Load the datasets and build the dataloaders.
    load_fed_dts: Load the datasets and build the dataloaders for the federated learning.
    _build_dataloaders: Build the dataloaders for the datasets.
    _load_torch_dts: Load the custom dataset from torchvision.
    _load_hf_dts: Load the dataset from the huggingface datasets library.
    _collate_fn_torch: Collate function to handle the data format of the returned datasets
        from torchvision.
    _collate_fn_hf: Collate function to handle the dict returned by the huggingface
        datasets library.
    _check_dts_type: Check if the dataset type is supported.

Constants:
    DTS_TYPES: The supported dataset types.
    CUSTOM_DTS: The custom datasets available.

Exceptions:
    ValueError: Raised when the dataset type is not supported.
    ValueError: Raised when the dataset is not available inside torchvision.datasets.
    ValueError: Raised when the dataset requires a huggingface token to be downloaded.
    ValueError: Raised when the data type is not supported.
    ValueError: Raised when the dataset type is not supported.

Author: Matteo Caligiuri
        Francesco Barbato
"""

from typing import Optional, Dict, List, Any, Union, Tuple, Callable
from pathlib import Path
from copy import deepcopy as dcp

import torch
import datasets
from datasets import DatasetDict

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose as Compose_v1
from torchvision.transforms.v2 import Compose, ToDtype
from torchvision import datasets as torch_datasets
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from modules.common import Logger, TqdmLogger
from modules.common.decorators import get_dts
from .partitioners import Partitioner, DirichletPartitioner


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


__all__ = ["load_dts", "load_fed_dts"]


# Constants
DTS_TYPES = {
    "hf": {"dts_fn": "_load_hf_dts", "collate_fn": "_collate_fn_hf"},
    "torch": {"dts_fn": "_load_torch_dts", "collate_fn": "_collate_fn_torch"},
}
CUSTOM_DTS = get_dts()


def load_dts(
    dts_info: DictConfig,
    batch_size: int,
    cache_dir: Optional[Union[Path, str]] = None,
    teacher_name: Optional[str] = None,
    transforms: Optional[Dict[str, Compose]] = None,
    generator: Optional[torch.Generator] = None,
    num_workers: int = 4,
    return_dataloader: bool = True,
) -> Tuple[Dict[str, Union[DataLoader, List[DataLoader], Dataset]], Dict[int, str]]:
    """
    Load the datasets and build the dataloaders.

    Args:
        dataset_info (DictConfig): The dataset information.
        batch_size (int): The batch size.
        cache_dir (Union[Path, str]): Path to caching directory.
        teacher_name (str): Name of the current teacher model class.
        transforms (Optional[Dict[str, Compose]], optional): The transformations to apply.
            Defaults to None. If None, the default transformation is applied (ToTensor(image)).
        generator (Optional[torch.Generator], optional): The random generator.
            Defaults to None.
        num_workers (int, optional): The number of workers to use. Defaults to 4.
        return_dataloader (bool, optional): If True, return the dataloaders.
            If False, return the datasets. Defaults to True.
    Returns:
        Tuple[Dict[str, Union[DataLoader, List[DataLoader], Dataset]], Dict[int, str]]:
            The dataloaders or dataset and
            the corresponding label mapping (index to text).
    """

    # Cast the OmegConf object to a dictionary
    dts_info = dict(dts_info)
    dts_type = dts_info.pop("type")

    # Define the laoding function
    dts_fn = _check_dts_type(dts_type)["dts_fn"]

    # Load the requested datasets
    dname = dts_info.pop("name")
    dts, label_mapping = dts_fn(
        dataset_name=dname,
        transforms=transforms,
        cache_dir=cache_dir,
        teacher_name=teacher_name,
        **dts_info,
    )
    dts["train"].name = dname
    dts["train"].num_classes = len(dts["train"].classes)

    # Build the dataloaders
    if return_dataloader:
        dataloaders = _build_dataloaders(
            dts,
            batch_size=batch_size,
            num_workers=num_workers,
            generator=generator,
            dts_type=dts_type,
        )
    else:
        dataloaders = dts

    return dataloaders, label_mapping


def load_fed_dts(
    dts_info: DictConfig,  # pylint: disable=unused-argument
    batch_size: int,
    transforms: Optional[Dict[str, Compose]] = None,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    partitioner: Optional[OmegaConf] = None,
    num_workers: int = 4,
    **kwargs: Dict[str, Any],
) -> callable:
    """
    Load the datasets and build the dataloaders.

    Args:
        dataset_info (DictConfig): The dataset information.
        batch_size (int): The batch size.
        transforms (Optional[Dict[str, Compose]], optional): The transformations to apply.
            Defaults to None. If None, the default transformation is applied (ToTensor(image)).
        seed (Optional[int], optional): The seed to use for the partitioning. Defaults to None.
        generator (Optional[torch.Generator], optional): The random generator.
            Defaults to None.
        partitioner (Optional[OmegaConf], optional): The partitioner to use.
            Defaults to DirichletPartitioner(num_clients=100, alpha=1).

    Returns:
        callable: The function to get the dataloader for a specific partition id.
    """

    # Cast the OmegConf object to a dictionary
    dts_info = dict(dts_info)
    dts_type = dts_info.pop("type")
    dts_name = dts_info.pop("name")

    # Define the laoding function
    dts_fn = _check_dts_type(dts_type)["dts_fn"]

    # Load the requested datasets and the label mapping
    dts, labels_mapping = dts_fn(
        dataset_name=dts_name,
        transforms={"train": transforms, "val": transforms},
        **dts_info,
    )

    # Partition the datasets
    # Define the partitiones
    if partitioner is None and seed is not None:
        partitioner_obj = DirichletPartitioner(
            num_partitions=100, alpha=1, shuffle=True, seed=seed, partition_by=1
        )
    elif partitioner is None and seed is None:
        partitioner_obj = DirichletPartitioner(
            num_partitions=100, alpha=1, shuffle=False, partition_by=1
        )
    else:
        partitioner_obj = instantiate(partitioner, shuffle=False, partition_by=1)

    partitioner = {k: dcp(partitioner_obj) for k in dts}
    del partitioner_obj

    # Load the correct dataset and labels to the partitioner
    for k, v in partitioner.items():
        v.dataset = dts[k]
        v.labels = labels_mapping

    # Build the dataloaders object
    return _FedDataLoader(
        partitioner=partitioner,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=generator,
        dts_type=dts_type,
    )


def _build_dataloaders(
    dts: DatasetDict,
    batch_size: int,
    num_workers: Optional[int] = 4,
    generator: Optional[torch.Generator] = None,
    dts_type: Optional[str] = "torch",
) -> Dict[str, DataLoader]:
    """
    Build the dataloaders for the datasets.

    Args:
        dts (DatasetDict): The datasets.
        batch_size (int): The batch size.
        num_workers (Optional[int], optional): The number of workers to use. Defaults to 4.
        generator (Optional[torch.Generator], optional): The random generator. Defaults to None.
        dts_type (Optional[str], optional): The dataset type. Defaults to "torch".
    Returns:
        Dict[str, DataLoader]: The dataloaders.
    """

    # Check the dataset type
    collate_fn = _check_dts_type(dts_type)["collate_fn"]

    dataloaders = {}
    for k, v in dts.items():
        if isinstance(v, Dataset):
            dataloaders[k] = DataLoader(
                v,
                batch_size=batch_size,
                num_workers=num_workers,
                generator=generator,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
            )
        elif isinstance(v, Partitioner):
            dataloaders[k] = DataLoader(
                v,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
            )
        elif isinstance(v, list):
            dataloaders[k] = [
                DataLoader(
                    i,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    generator=generator,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=collate_fn,
                )
                for i in v
            ]
        else:
            raise ValueError(f"Dataset data type not supported: {type(v)}")

    return dataloaders


def _load_torch_dts(
    dataset_name: str,
    data_dir: Union[Path, str],
    cache_dir: Optional[Union[Path, str]] = None,
    teacher_name: Optional[str] = None,
    transforms: Optional[Dict[str, Compose]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Dataset], Dict[int, str]]:
    """
    Load the custom dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        data_dir (Union[Path, str]): The directory where the dataset is stored.
        cache_dir (Union[Path, str]): Path to caching directory.
        teacher_name (str): Name of the current teacher model class.
        transforms (Optional[Dict[str, Compose]], optional): The transformations to apply.
            Defaults to None. If None, the default transformation is applied (ToTensor(image)).
        **kwargs: Additional keyword arguments to pass to the dataset constructor.
    Returns:
        Tuple[Dict[str, torch.Dataset], Dict[int, str]]: The datasets and the corresponding
    """

    # Parse the transforms
    if transforms is None:
        transforms = {"train": torch.nn.Identity(), "val": torch.nn.Identity()}

    # Cast the data_dir to Path if it is a string
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

    # Check if the dataset is available
    if dataset_name not in torch_datasets.__dict__ and dataset_name not in CUSTOM_DTS:
        raise ValueError(
            f"The dataset {dataset_name} is not available inside torchvision.datasets."
        )

    # Define the dataset class to be used
    available_dts = torch_datasets.__dict__
    for c in CUSTOM_DTS:
        available_dts[c] = CUSTOM_DTS[c]
    dts_cls = available_dts[dataset_name]["dts"]

    # Load the dataset
    if dataset_name != "ImageFolder":
        dts = {
            k: dts_cls(root=data_dir, split=k, transform=transforms[k], **kwargs)
            for k in kwargs.pop("splits")
        }
    else:
        splits = kwargs.pop("splits")
        dts = {
            k: dts_cls(root=data_dir / splits[k], transform=transforms[k], **kwargs)
            for k in splits
        }

    # load the training set embeddings, if they exist
    for split in dts:
        dts[split].embeddings = None
        if cache_dir is not None and teacher_name is not None:
            embed_folder = Path(cache_dir) / "embeddings" / teacher_name / split
            embed_folder.mkdir(parents=True, exist_ok=True)
            embed_path = embed_folder / f"{dts[split].__class__.__name__.lower()}.pth"
            if embed_path.exists():
                logger.info(
                    "Loading %s precomputed embeddings from %s",
                    teacher_name,
                    embed_path,
                )
                dts[split].embeddings = torch.load(
                    embed_path, "cpu", weights_only=False
                )
                logger.info("Embeddings loaded")
            else:
                logger.info("No precomputed embeddings found for split %s", split)

    # Load the label mapping
    label_mapping = {
        k: v[0] if isinstance(v, tuple) else v
        for k, v in enumerate(dts["train"].classes)
    }

    return dts, label_mapping


def _load_hf_dts(
    dataset_name: str,
    transforms: Optional[Dict[str, Compose]] = None,
    **kwargs,
) -> Tuple[Dict[str, datasets.Dataset], Dict[int, str]]:
    """
    Load the dataset from the huggingface datasets library, apply the transformations.

    Args:
        dataset_name (str): The name of the dataset to load.
        transforms (Optional[Dict[str, Compose]], optional): The transformations to apply.
            Defaults to None. If None, the default transformation is applied (ToTensor(image)).
        **kwargs: Additional keyword arguments to pass to the dataset constructor.
    Returns:
        Tuple[Dict[str, datasets.Dataset], Dict[int, str]]: The datasets and the corresponding
            label mapping (index to text).
    """

    # Parse the cache_dir
    cache_dir = Path(kwargs.pop("data_dir"))

    # Check if the token is provided
    try:
        token = kwargs.pop("token")
    except KeyError as e:
        raise ValueError(
            f"The dataset {dataset_name} requires a huggingface token to be downloaded."
        ) from e

    # Define the splits to be selected
    try:
        splits = kwargs.pop("splits")
    except KeyError:
        splits = ["train", "validation", "test"]
    # Map the splits name to the correct format
    if isinstance(splits, str):
        splits = [splits]

    # Load the dataset from the huggingface datasets library
    dts = {
        k: datasets.load_dataset(
            path=dataset_name, cache_dir=cache_dir, token=token, split=k
        )
        for k in splits
    }

    # Process the transformations
    # Default transformation
    if transforms is None:
        transforms = Compose([ToDtype(dtype=torch.float32)])

    # Custom transformation
    if isinstance(transforms, Compose) or isinstance(transforms, Compose_v1):
        # Single transformation
        transforms = {k: _Transform(transform=transforms) for k in dts.keys()}
    else:
        # Different transformation for each split
        transforms = {k: _Transform(transform=v) for k, v in transforms.items()}

    # Apply the transformations
    for k, v in dts.items():
        v.set_transform(
            transform=transforms[k], columns=["image"], output_all_columns=True
        )

    # Map the validation labels (if present) to val
    if "validation" in dts:
        dts["val"] = dts.pop("validation")

    # Load the label mapping
    label_mapping = {
        k: v.split(", ")[0]
        for k, v in enumerate(dts[list(dts.keys())[0]].features["label"].names)
    }

    return dts, label_mapping


def _collate_fn_torch(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle the data format of the returned datasets from torchvision.

    Args:
        batch (List[torch.Tensor, torch.Tensor]): The batch to collate.
    Returns:
        Dict[str, torch.Tensor]: The collated batch.
    """

    if isinstance(batch[0][0], dict):
        c_images = torch.stack([x[0]["client"][0] for x in batch])
        s_images = torch.stack([x[0]["server"][0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        embeddings = (
            torch.stack([x[2] for x in batch])
            if len(batch[0]) > 2 and batch[0][2] is not None
            else None
        )
        torch_batch = {
            "c_image": c_images,
            "s_image": s_images,
            "label": labels,
            "embeddings": embeddings,
        }
    else:
        iamges = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        embeddings = (
            torch.stack([x[2] for x in batch])
            if len(batch[0]) > 2 and batch[0][2] is not None
            else None
        )
        torch_batch = {"image": iamges, "label": labels, "embeddings": embeddings}

    return torch_batch


def _collate_fn_hf(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle the dict returned by the huggingface datasets library.

    Args:
        batch (List[Dict[str, Any]]): The batch to collate.
    Returns:
        Dict[str, torch.Tensor]: The collated batch.
    """

    torch_batch = {}
    for k in batch[0].keys():
        if isinstance(batch[0][k], torch.Tensor):
            data = torch.stack([x[k] for x in batch])
        elif isinstance(batch[0][k], int):
            data = torch.tensor([x[k] for x in batch])
        else:
            raise ValueError(f"Data type not supported: {type(batch[0][k])}")
        torch_batch[k] = data
    return torch_batch


def _check_dts_type(dts_type: str) -> Dict[str, Callable]:
    """
    Check if the dataset type is supported.

    Args:
        dts_type (str): The dataset type.
    Returns:
        Dict[str, Callable]: A dict containing the corresponding functions to load the datasetand
            and to collate the batch.
    """

    if dts_type not in DTS_TYPES:
        raise ValueError(
            f"Dataset type {dts_type} not supported,"
            + f"supported types are {', '.join(list(DTS_TYPES.keys()))}."
        )

    return {
        "dts_fn": globals()[DTS_TYPES[dts_type]["dts_fn"]],
        "collate_fn": globals()[DTS_TYPES[dts_type]["collate_fn"]],
    }


class _Transform:
    """
    Apply the transformations to the data.
    Wrapper of the Compose class from torchvision.transforms that allows
    to handle the dict returned by the huggingface datasets library.

    Args:
        transform (Compose): The transformations to apply.
    """

    def __init__(self, transform: Compose):
        self.transform = transform

    def __call__(self, x: Dict[str, List[Any]]) -> Dict[str, List[torch.Tensor]]:
        """
        Apply the transformations and move to device.

        Args:
            x (Dict[str, List[Any]]): The data to transform.
        Returns:
            Dict[str, List[torch.Tensor]]: The transformed data.
        """

        for k, v in x.items():
            x[k] = [self.transform(i) for i in v]
        return x


class _FedDataLoader:
    """
    This class is responsible for loading the federated datasets.
    given a partitioner object for each split (as a dict) when called with a specific id
    returns the dataloader for that partitioner.
    When called with a specific id, it returns the dataloader for that partition.

    Args:
        partitioner (Dict[str, Partitioner]): The partitioners.
        batch_size (int): The batch size.
        num_workers (Optional[int], optional): The number of workers to use. Defaults to 4.
        generator (Optional[torch.Generator], optional): The random generator. Defaults to None.
        dts_type (Optional[str], optional): The dataset type. Defaults to "torch".
    """

    def __init__(
        self,
        partitioner: Dict[str, Partitioner],
        batch_size: int,
        num_workers: Optional[int] = 4,
        generator: Optional[torch.Generator] = None,
        dts_type: Optional[str] = "torch",
    ) -> None:
        self.partitioners = partitioner
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = generator
        self.dts_type = dts_type

    def __call__(
        self, id: Union[int, str]  # pylint: disable=redefined-builtin
    ) -> DataLoader:
        """
        Call the object with a specific id to get the dataloader for that partition.

        Args:
            id (Union[int, str]): The partition id.
        Returns:
            DataLoader: The dataloader for the partition.
        """

        # Parse the id
        if isinstance(id, str):
            id = int(id)

        return _build_dataloaders(
            dts={
                k: v.load_partition(partition_id=id)
                for k, v in self.partitioners.items()
            },
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.generator,
            dts_type=self.dts_type,
        )
