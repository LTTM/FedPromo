"""
This script is responsible for evaluating models across
multiple datasets in a federated learning setup.
It includes functionality for loading datasets, performing inference,
and computing metrics.

Classes:
    ConcatDataset: A dataset class that concatenates multiple datasets.

Functions:
    loader: Loads all the provided checkpoints and concatenates the heads.
    create_compose: Creates a transform module for preprocessing images.
    instantiate_models: Instantiates and loads models with the provided checkpoints.
    load_yaml_to_dict: Loads a YAML file into a Python dictionary.
    load_datasets: Loads datasets and returns a DataLoader for evaluation.
    inference: Performs inference on a dataset using a model and computes metrics.

Constants:
    DATA_FOLDER: Path to the data folder.
    BATCH_SIZE: Batch size for DataLoader.
    NUM_WORKERS: Number of workers for DataLoader.
    USE_CDB: Boolean flag to indicate whether to use CBD checkpoints.
    NAS_PATH: Path to the NAS storage.
    PROJECT_FOLDER: Path to the project folder.
    DTS_ROOT_DIR: Root directory for datasets.
    DTS_CONFIG_PATH: Path to the dataset configuration files.
    DATA: Dictionary containing dataset and model information.

Author: Matteo Caligiuri
        Francesco Barbato
"""

from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple
from collections import OrderedDict
import bisect
import os
import sys

from tqdm import tqdm
import yaml
import torch
from torchvision.transforms import v2 as T
from omegaconf import DictConfig
from matplotlib import pyplot as plt, colors as plt_colors

# Allow running this script directly from extras/scripts/ while still
# importing the top-level `modules` package from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.datasets import load_dts
from modules.models import instantiate_dinov2, SimpleClassifier


# General constants
DATA_FOLDER = Path("./data")
OUTPUT_FOLDER = DATA_FOLDER / "multidomain_logs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
BATCH_SIZE = 1
NUM_WORKERS = 8
SERVER_MODEL = "dinov2_vitl14_reg"  # The server model to use for evaluation
USE_CDB = True  # Set to True if you want to use CBD, otherwise set to False
DTS_TO_LOAD = "all"  # Specify the datasets to load, or "all" for all datasets
NORMALZIE_WEIGHTS = False  # Set to True if you want to normalize the head weights
NORMALIZE_FEATS = False  # Set to True if you want to normalize the features

# Dataset and model information
CKPT_PATH = Path("./data/checkpoints/final")
DTS_ROOT_DIR = DATA_FOLDER / "datasets"
DTS_CONFIG_PATH = Path("./conf/federated/dts/")
DATA = {
    "compcars": {
        "ckpt": Path(CKPT_PATH / "compcars.pth"),
        "ckpt_no_cbd": Path(CKPT_PATH / "compcars_noCDB.pth"),
        "dts_path": Path(DTS_ROOT_DIR / "CompCars_torch"),
        "n_cls": 431,
        "ref_color": "#FF0000",
        "name": "CompCars",
    },
    "uecfood256": {
        "ckpt": Path(CKPT_PATH / "food.pth"),
        "ckpt_no_cbd": Path(CKPT_PATH / "food_noCDB.pth"),
        "dts_path": Path(DTS_ROOT_DIR / "UECFOOD256_torch"),
        "conf_name": "uecfood_256",
        "n_cls": 256,
        "ref_color": "#c46409",
        "name": "UECFOOD256",
    },
    "nabirds": {
        "ckpt": Path(CKPT_PATH / "birds.pth"),
        "ckpt_no_cbd": Path(CKPT_PATH / "birds_noCDB.pth"),
        "dts_path": Path(DTS_ROOT_DIR / "NABirds_torch"),
        "n_cls": 555,
        "ref_color": "#00ff37",
        "name": "NABirds",
    },
    "military_aircraft": {
        "ckpt": Path(CKPT_PATH / "military.pth"),
        "ckpt_no_cbd": Path(CKPT_PATH / "military_noCDB.pth"),
        "dts_path": Path(DTS_ROOT_DIR / "MilitaryAircraft_torch"),
        "n_cls": 80,
        "ref_color": "#000595",
        "name": "Military\nAircraft",
    },
    # "stanford_dogs": {
    #     "ckpt": Path(CKPT_PATH / "dogs.pth"),
    #     "ckpt_no_cbd": Path(CKPT_PATH / "dogs_noCDB.pth"),
    #     "dts_path": Path(DTS_ROOT_DIR / "StanfordDogs_torch"),
    #     "n_cls": 120,
    #     "ref_color": "#ff00c8",
    #     "name": "Stanford\nDogs",
    # },
    "oxford_pets": {
        "ckpt": Path(CKPT_PATH / "pets.pth"),
        "ckpt_no_cbd": Path(CKPT_PATH / "pets_noCDB.pth"),
        "dts_path": Path(DTS_ROOT_DIR / "OxfordPets_torch"),
        "n_cls": 37,
        "ref_color": "#ff00c8",
        "name": "Oxford\nPets",
    },
}


class NormLayer(torch.nn.Module):
    """
    A normalization layer that normalizes the input tensor.

    Args:
        dim (int): The dimension along which to normalize the input tensor.
            Defaults to 1 (i.e., normalize along the feature dimension).

    Returns:
        torch.Tensor: The normalized tensor.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor along the specified dimension.
        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """

        return x / torch.norm(x, dim=self.dim, keepdim=True)


class ConcatDataset(torch.utils.data.ConcatDataset):
    """
    A dataset that concatenates multiple datasets.
    """

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


def _collate_fn_torch(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle the data format of the returned datasets from torchvision.

    Args:
        batch (List[torch.Tensor, torch.Tensor]): The batch to collate.
    Returns:
        Dict[str, torch.Tensor]: The collated batch.
    """

    if len(batch[0]) == 2:
        iamges = torch.stack([x[0][0] for x in batch])
        labels = torch.tensor([x[0][1] for x in batch])
        embeddings = (
            torch.stack([x[0][2] for x in batch])
            if len(batch[0][0]) > 2 and batch[0][0][2] is not None
            else None
        )
        dts_idx = torch.tensor([x[1] for x in batch])
    else:
        iamges = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        embeddings = (
            torch.stack([x[2] for x in batch])
            if len(batch[0]) > 2 and batch[0][2] is not None
            else None
        )
        dts_idx = torch.tensor(
            [0 for _ in batch]
        )  # Default to 0 if no dataset index is provided
    torch_batch = {
        "image": iamges,
        "label": labels,
        "embeddings": embeddings,
        "dts_idx": dts_idx,
    }

    return torch_batch


def create_compose() -> torch.nn.Module:
    """
    Function to create a transform module.

    Returns:
        torch.nn.Module: The composed transform.
    """
    # Define the transformations for preprocessing images
    return T.Compose(
        [
            T.PILToTensor(),
            T.RGB(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
            T.Resize((224, 224)),
            T.CenterCrop((224, 224)),
        ]
    )


def plt_self_sim(
    model_weights: torch.Tensor,
    out_path: Union[str, Path],
    cmap: Optional[str] = "viridis",
) -> None:
    """
    Function to plot the self-similarity matrix of the model weights.

    Args:
        model_weights (torch.Tensor): The model weights to compute the self-similarity matrix.
        out_path (Union[str, Path]): The path to save the plot.
        cmap: (str, optional): The colormap to use for the plot.
            Defaults to "viridis".

    Returns:
        None
    """

    # Parse the output path, if needed
    if isinstance(out_path, str):
        out_path = Path(out_path)

    # Check that the colormap is valid
    if cmap not in plt.colormaps():
        raise ValueError(
            f"Invalid colormap: {cmap}. Available colormaps: {plt.colormaps()}"
        )

    # Create the figure and save it
    fig, ax = plt.subplots(figsize=(25, 25))
    w_norm = model_weights / torch.norm(model_weights, dim=1, keepdim=True)
    similarity = w_norm @ w_norm.T
    ax.imshow(similarity.cpu().numpy(), cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Model Weights", fontsize=20)
    ax.set_ylabel("Model Weights", fontsize=20)
    ax.grid(False)  # Disable grid lines
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def create_rgb_matrix(
    confmat: torch.Tensor, types: List[str]
) -> Tuple[torch.Tensor, List[int]]:
    """
    Function to create a RGB matrix from the confusion matrix.

    Args:
        confmat (torch.Tensor): The confusion matrix to convert.
        types (List[str]): The list of dataset names.

    Returns:
        Tuple[torch.Tensor, List[int]]: The RGB matrix and the class intervals.
    """

    # Load all the colors from the DATA dictionary and convert in RGB format
    colors = [plt_colors.to_rgb(DATA[dts]["ref_color"]) for dts in types]

    # Extract the correct range of the confusion matrix
    cls_intervals = [
        sum(DATA[dts]["n_cls"] for dts in types[:i]) for i in range(len(types) + 1)
    ]

    # Create the RGBA matrix
    rgb_matrix = torch.zeros(
        confmat.shape[0],
        confmat.shape[1],
        4,  # RGBA format
        device=confmat.device,
        dtype=torch.float32,
    )

    # Fill the RGB matrix with the colors
    colors = torch.tensor(colors, device=confmat.device, dtype=torch.float32)
    cls_intervals = torch.tensor(
        cls_intervals, device=confmat.device, dtype=torch.float32
    )
    cols = (
        torch.arange(0, confmat.shape[0], device=confmat.device)
        .unsqueeze(1)
        .repeat(1, confmat.shape[1])
    )
    rows = (
        torch.arange(0, confmat.shape[1], device=confmat.device)
        .unsqueeze(0)
        .repeat(confmat.shape[0], 1)
    )

    label_dts_idxs = torch.bucketize(cols, cls_intervals) - 1
    pred_dts_idxs = torch.bucketize(rows, cls_intervals) - 1

    label_color = colors[label_dts_idxs]
    pred_color = colors[pred_dts_idxs]

    cell_color = torch.sqrt((label_color**2 + pred_color**2) / 2)
    cell_alpha = 0.1 + 0.9 * confmat

    rgb_matrix[..., :3] = cell_color
    rgb_matrix[..., 3] = cell_alpha

    return rgb_matrix, cls_intervals.tolist()


def plt_conf_matrix(
    confmat: torch.Tensor,
    out_path: Union[str, Path],
    types: Union[str, List[str]] = "all",
) -> None:
    """
    Function to plot the confusion matrix.

    Args:
        confmat (torch.Tensor): The confusion matrix to plot.
        out_path (Union[str, Path]): The path to save the plot.
        types (Union[str, List[str]]): The type of dataset to plot.
            If "all", plot all datasets.
            Defaults to "all".

    Returns:
        None
    """

    # Parse types argument
    if isinstance(types, str) and types == "all":
        types = list(DATA.keys())
    elif isinstance(types, str):
        raise ValueError(
            f"Invalid type: {types}. If you want to use all datasets, use 'all' as the type."
        )

    # Parse the output path, if needed
    if isinstance(out_path, str):
        out_path = Path(out_path)

    # Normalize the conf matrix by rows
    confmat /= torch.clamp_min(confmat.sum(dim=1, keepdim=True), 1)

    # Define a custom colormap for each datast
    rgb_matrix, cls_intervals = create_rgb_matrix(confmat, types)

    # Define the line width of the dashed separator lines
    linewidth = 2

    # Create the figure
    fig, ax = plt.subplots(figsize=(25, 25))
    ax.imshow(rgb_matrix.cpu().numpy())

    # Add the dashed separator lines
    for pos in cls_intervals[1:-1]:
        pos -= 0.5
        ax.plot(
            [-0.5, rgb_matrix.shape[0] - 0.5],
            [pos, pos],
            color="black",
            linewidth=linewidth,
            linestyle="--",
        )
        ax.plot(
            [pos, pos],
            [-0.5, rgb_matrix.shape[0] - 0.5],
            color="black",
            linewidth=linewidth,
            linestyle="--",
        )
    ax.set_xlim(-0.5, rgb_matrix.shape[0] - 0.5)
    ax.set_ylim(rgb_matrix.shape[0] - 0.5, -0.5)

    # Customize the tics to show the dts names instead of the class indices
    n_classes = [DATA[dts]["n_cls"] for dts in types]
    ax.set_xticks(
        [i + c / 2 - 0.5 for i, c in zip(cls_intervals[:-1], n_classes)],
        labels=[DATA[dts]["name"] for dts in types],
        fontsize=18,
    )
    ax.set_yticks(
        [i + c / 2 - 0.5 for i, c in zip(cls_intervals[:-1], n_classes)],
        labels=[DATA[dts]["name"] for dts in types],
        rotation=90,
        va="center",
        ma="center",
        fontsize=18,
    )
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=True,
        labelbottom=True,
    )

    # Save the figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def loader(
    cdb: bool = True, types: Union[str, List[str]] = "all", normalize: bool = False
) -> OrderedDict:
    """
    Function to load all the provided checkpoints.

    Args:
        cdb (bool): If True, load the checkpoints with CBD,
            otherwise load the checkpoints without CBD.
        types (Union[str, List[str]]): The type of dataset to load.
            If "all", load all the datasets.
        normalize (bool): If True, normalize the head weights.

    Returns:
        OrderedDict: head loaded checkpoints.
    """
    if isinstance(types, str) and types == "all":
        types = list(DATA.keys())
    elif isinstance(types, str):
        types = [types]

    heads = []

    # Load all the datasets
    for dataset_name in types:
        if dataset_name not in DATA:
            raise ValueError(f"Dataset {dataset_name} not found in DATA dictionary.")

        if "ckpt_no_cbd" not in DATA[dataset_name] and not cdb:
            raise ValueError(
                f"Dataset {dataset_name} does not have a non-CBD checkpoint."
            )

        checkpoint_path = (
            DATA[dataset_name]["ckpt"] if cdb else DATA[dataset_name]["ckpt_no_cbd"]
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Load correspondent data
        head_weights = state_dict["head"]["fcn.0.weight"][: DATA[dataset_name]["n_cls"]]

        # Normalize the head weights
        if normalize:
            scale = head_weights.norm(dim=1, keepdim=True)
            head_weights = head_weights / scale

        heads.append(head_weights)

    # Concatenate the heads
    concatenated_head = torch.cat(heads, dim=0)

    # Compute and save the self-similarity matrix
    plt_self_sim(
        model_weights=concatenated_head,
        out_path=OUTPUT_FOLDER / "weights_similarity.pdf",
        cmap="viridis",
    )

    return OrderedDict(
        {
            "fcn.0.weight": concatenated_head,
        }
    )


def instantiate_models(
    checkpoint: OrderedDict,
    torch_device: torch.device,
    normalize: bool = False,
    server_only: bool = False,
) -> torch.nn.Module:
    """
    Function to instantiate the models.

    Args:
        checkpoint (OrderedDict): The checkpoint of the head.
        torch_device (torch.device): The device to load the model on.
        normalize (bool): If True, apply normalization to the features.
        server_only (bool): If True, only instantiate the server model without the head.

    Returns:
        torch.nn.Module: The instantiated model with the head.
    """

    # Instantiate the MobileNetv3 model
    net_model = instantiate_dinov2(
        pretrained_model=SERVER_MODEL,
        data_folder=DATA_FOLDER,
        device=torch_device,
    )

    # If server_only is True, return the model without the head
    if server_only:
        return net_model.to(torch_device)

    # Extract the number of classes from the checkpoint and the features size
    n_classes, features_size = checkpoint["fcn.0.weight"].shape

    # Instantiate the head
    head = SimpleClassifier(
        num_layers=1,
        dropouts=None,
        features=features_size,
        num_classes=n_classes,
    )

    # Load the state dict into the model and head
    head.load_state_dict(checkpoint)

    # Define the feats norm layer if needed
    if normalize:
        feats_norm = NormLayer(dim=1)
    else:
        feats_norm = torch.nn.Identity()

    # Create a single model that contains both the model and the head
    full_model = torch.nn.Sequential(
        net_model,
        feats_norm,
        head,
    )

    return full_model.to(torch_device)


def load_yaml_to_dict(yaml_path):
    """
    Load a YAML file into a Python dictionary.

    Args:
        yaml_path (Path): Path to the YAML file.

    Returns:
        dict: Parsed YAML content as a dictionary.
    """
    with open(yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_datasets(
    data_dict: Dict[str, Dict[str, Union[str, Path]]],
    types: Union[str, List[str]] = "all",
    split: str = "val",
) -> List[torch.utils.data.DataLoader]:
    """
    Function to load the datasets.

    Args:
        data_dict (Dict[str, Dict[str, Union[str, Path]]]): Dictionary
            containing dataset information.
            defaults to DATA.
        types (Union[str, List[str]]): The type of dataset to load. If "all", load all the datasets.
        split (str): The split of the dataset to load. Defaults to "val".
            Defaults to "val".

    Returns:
        List[torch.utils.data.DataLoader]: The list of data loaders for the datasets.
    """

    if isinstance(types, str) and types == "all":
        types = list(data_dict.keys())
    elif isinstance(types, str):
        types = [types]

    dts = []
    for dts_name in tqdm(
        types, desc="Loading datasets", unit="dataset", colour="green"
    ):
        if dts_name not in data_dict:
            raise ValueError(f"Dataset {dts_name} not found in DATA dictionary.")

        # Load the correspondent dataset info DictConfig
        if "conf_name" in data_dict[dts_name]:
            dataset_info_path = (
                DTS_CONFIG_PATH / f"{data_dict[dts_name]['conf_name']}.yaml"
            )
        else:
            dataset_info_path = DTS_CONFIG_PATH / f"{dts_name}.yaml"

        cfg_file = load_yaml_to_dict(dataset_info_path)
        cfg_file["data_dir"] = str(
            data_dict[dts_name]["dts_path"]
            # DTS_ROOT_DIR / cfg_file["data_dir"].replace("${dts_root_dir}datasets/", "")
        )

        dataset_info = DictConfig(cfg_file)

        # Load the dataloader
        dataset, _ = load_dts(
            dts_info=dataset_info,
            batch_size=BATCH_SIZE,
            cache_dir=DATA_FOLDER,
            teacher_name=None,
            transforms={"train": create_compose(), "val": create_compose()},
            generator=None,
            num_workers=NUM_WORKERS,
            return_dataloader=False,
        )

        # Remove the train split and keep only the test or validation split
        if split == "val":
            if "test" in dataset:
                test_dataset = dataset["test"]
            elif "val" in dataset:
                test_dataset = dataset["val"]
            else:
                raise ValueError(
                    f"No test or validation split found in dataset {dts_name}."
                )
        else:
            if "train" in dataset:
                test_dataset = dataset["train"]
            else:
                raise ValueError(f"No train split found in dataset {dts_name}.")

        # Remove the dataloader variable to free memory
        del dataset

        dts.append(test_dataset)

    # Compose the dataset in a unique DataLoader
    if len(dts) > 1:
        # If the dataset is a list, concatenate them
        test_dataset = ConcatDataset(dts)
    else:
        # If the dataset is a single dataset, keep it as is
        test_dataset = dts[0]

    # Create the DataLoader for the test dataset
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=_collate_fn_torch,
        drop_last=True,
    )

    return test_dataloader


@torch.inference_mode
def inference(
    data_loader: torch.utils.data.DataLoader,
    model_instance: torch.nn.Module,
    device_instance: torch.device,
    types: Optional[Union[str, List[str]]] = "all",
):
    """
    Function to perform inference on a dataset using a model.

    Args:
        data_loader (torch.utils.data.DataLoader): The dataset to perform inference on.
        model_instance (torch.nn.Module): The model to use for inference.
        device_instance (torch.device): The device to perform inference on.
        types (Optional[Union[str, List[str]]]): list of datasets to consider.

    Returns:
        dict: Metrics including accuracy@1 and accuracy@5 for each dataset.
    """
    print("🔄 Initializing inference metrics...")

    # Parse the types argument
    if isinstance(types, str) and types == "all":
        types = list(DATA)
    elif isinstance(types, str) and types in DATA:
        types = [types]
    elif isinstance(types, str):
        raise ValueError(
            f"Invalid type: {types}. If you want to use all datasets, use 'all' as the type."
        )

    # Define a mapping from dataset index to rescale factor
    rescale_map = [0]
    for i in range(1, len(types)):
        rescale_map.append(rescale_map[i - 1] + DATA[types[i - 1]]["n_cls"])

    # Extract the total number of classes and initialize an empty confusion matrix
    n_classes = rescale_map[-1] + DATA[types[-1]]["n_cls"]
    confmat = torch.zeros(n_classes, n_classes, device=device_instance)

    # Initialize metrics storage
    top_ks = {"total": []}
    labels_acc = {"total": []}
    for dataset_name in types:
        top_ks[dataset_name] = []
        labels_acc[dataset_name] = []

    # Perform inference with progress bar
    for batch in tqdm(
        data_loader, desc="Inference Progress", unit="batch", colour="green"
    ):
        # Move the data to the device
        images, labels, dataset_indices = (
            batch["image"],
            batch["label"],
            batch["dts_idx"],
        )
        images, labels = images.to(device_instance), labels.to(device_instance)

        # Perform inference
        outputs = model_instance(images)

        # Rescale the labels
        for i in range(labels.shape[0]):
            labels[i] += rescale_map[dataset_indices[i].item()]

        # Get the class predictions
        _, top_k = torch.topk(outputs, k=5, dim=1)

        # Store the prediction and labels
        top_ks["total"].append(top_k.detach())
        labels_acc["total"].append(labels.detach())

        # Update the confusion matrix
        confmat[labels, top_k[:, 0]] += 1

        for dataset_index in dataset_indices:
            dataset_name = types[dataset_index.item()]
            top_ks[dataset_name].append(top_k.detach())
            labels_acc[dataset_name].append(labels.detach())

    # Save the confusion matrix
    print("💾 Saving confusion matrix...")
    plt_conf_matrix(
        confmat=confmat,
        out_path=OUTPUT_FOLDER / "confusion_matrix.pdf",
        types=types,  # Pass the types to the plotting function
    )

    print("📊 Computing final metrics...")

    # Define the metrics dict
    out_metrics = {
        "accuracy-1": 0.0,
        "accuracy-5": 0.0,
    }
    for dataset_name in types:
        out_metrics[f"{dataset_name}_accuracy-1"] = 0.0
        out_metrics[f"{dataset_name}_accuracy-5"] = 0.0

    # Compute the final accuracies
    for dataset_key in top_ks:
        top_ks[dataset_key] = torch.cat(top_ks[dataset_key], dim=0)
        labels_acc[dataset_key] = torch.cat(labels_acc[dataset_key], dim=0)

    out_metrics["accuracy-1"] = torch.mean(
        1.0
        * (
            top_ks["total"][:, 0]  # pylint: disable=invalid-sequence-index
            == labels_acc["total"]
        )
    ).item()
    out_metrics["accuracy-5"] = torch.mean(
        1.0
        * torch.any(
            top_ks["total"]
            == labels_acc["total"].unsqueeze(1),  # pylint: disable=no-member
            dim=1,
        )
    ).item()

    for dataset_name in types:
        out_metrics[f"{dataset_name}_accuracy-1"] = torch.mean(
            1.0
            * (
                top_ks[dataset_name][:, 0] == labels_acc[dataset_name]
            )  # pylint: disable=invalid-sequence-index
        ).item()
        out_metrics[f"{dataset_name}_accuracy-5"] = torch.mean(
            1.0
            * torch.any(
                top_ks[dataset_name] == labels_acc[dataset_name].unsqueeze(1), dim=1
            )
        ).item()

    return out_metrics


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load all the checkpoints and concatenate the heads
    print("📦 Loading checkpoints and concatenating heads...")
    ckpt = loader(cdb=USE_CDB, types=DTS_TO_LOAD, normalize=NORMALZIE_WEIGHTS)
    print("✅ Checkpoints loaded successfully")

    # Instantiate and load the models
    print("🤖 Instantiating and loading the model...")
    model = instantiate_models(ckpt, torch_device=device, normalize=NORMALIZE_FEATS)
    print("✅ Model instantiated and loaded successfully")

    # Load the datasets
    print("📂 Loading datasets...")
    dataloader = load_datasets(data_dict=DATA, types=DTS_TO_LOAD)
    print("✅ Datasets loaded successfully")

    # Perform inference
    print("🔄 Starting inference...")
    metrics = inference(dataloader, model, device, types=DTS_TO_LOAD)
    print("✨ Inference completed")

    # Log the metrics
    print("================= Inference Metrics =================")
    print(f"🧪 CDB Enabled: {'Yes' if USE_CDB else 'No'}")
    print(f"⚖️  Weights Normalization: {'Yes' if NORMALZIE_WEIGHTS else 'No'}")
    print(f"🔧 Feature Normalization: {'Yes' if NORMALIZE_FEATS else 'No'}")
    for key, value in metrics.items():
        print(f"📊 {key:<30}: {value*100:>6.2f}%")
    print("=====================================================")
