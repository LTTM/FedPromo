"""
Script for extracting and caching embeddings from pre-trained DINOv2 models.

This module generates feature embeddings for specified datasets using a pre-trained
DINOv2 model and caches them for later use in federated learning experiments.
The embeddings are computed on the training split and saved as PyTorch tensors.

Classes:
    None

Functions:
    None

Constants:
    SPLIT_TO_LOAD: The dataset split to process for embedding extraction.
    DTS_TO_LOAD: The target dataset name for embedding generation.
    SERVER_MODEL: The DINOv2 model variant to use for feature extraction.
    DATA_FOLDER: Base directory for storing processed data and embeddings.
    DTS_ROOT_DIR: Root directory containing the raw dataset files.
    EMBEDDINGS_FOLDER: Output directory for storing the generated embeddings.
    DATA_DICT: Dictionary mapping dataset names to their respective paths.

Exceptions:
    None

Author: Matteo Caligiuri
        Francesco Barbato
"""

from pathlib import Path

from tqdm import tqdm
import torch

from multi_domain_eval import (
    instantiate_models,
    load_datasets,
    DATA,
    DTS_ROOT_DIR as OLD_DTS_ROOT_DIR,
)

# General constants
SPLIT_TO_LOAD = "train"  # Specify the split to load
DTS_TO_LOAD = "stanfordcars"  # Specify the datasets to load
SERVER_MODEL = "dinov2_vitl14_reg"  # The server model to use for evaluation
DATA_FOLDER = Path("./data")
DTS_ROOT_DIR = Path(DATA_FOLDER / "datasets")
EMBEDDINGS_FOLDER = DATA_FOLDER / f"embeddings/dinov2classifier/{SPLIT_TO_LOAD}"

# Create embeddings output directory if it doesn't exist
EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)

# Replace the dataset root dir for all the datasets
# This updates the paths from the old configuration to the new dataset location
DATA_DICT = {
    key: {
        "dts_path": Path(
            str(value["dts_path"]).replace(str(OLD_DTS_ROOT_DIR), str(DTS_ROOT_DIR))
        )
    }
    for key, value in DATA.items()
}

# Pre-trained dataset definitions
# Define dataset paths for various computer vision datasets
DATA_DICT["stanfordcars"] = {
    "dts_path": Path(DTS_ROOT_DIR / "StanfordCars_torch"),
}
DATA_DICT["food101"] = {
    "dts_path": Path(DTS_ROOT_DIR / "Food101_torch"),
}
DATA_DICT["cub200"] = {
    "dts_path": Path(DTS_ROOT_DIR / "CUB200_torch"),
}
DATA_DICT["fgvcaircraft"] = {
    "dts_path": Path(DTS_ROOT_DIR / "FGVCAircraft_torch"),
}
DATA_DICT["stanforddogs"] = {
    "dts_path": Path(DTS_ROOT_DIR / "StanfordDogs_torch"),
}


if __name__ == "__main__":
    # Initialize computation device
    # Use CUDA if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting embedding extraction using device: {device}")

    # Load and initialize the DINOv2 model
    # The model is instantiated with server-only configuration
    print(f"📦 Loading {SERVER_MODEL} model...")
    dino = instantiate_models(
        checkpoint={},
        torch_device=device,
        server_only=True,
    )
    # Set model to evaluation mode for inference
    dino.eval()
    print("✅ Model loaded successfully")

    # Load the target dataset
    print(f"📂 Loading {DTS_TO_LOAD} dataset ({SPLIT_TO_LOAD} split)...")
    tset = load_datasets(data_dict=DATA_DICT, types=DTS_TO_LOAD, split=SPLIT_TO_LOAD).dataset
    print(f"📊 Dataset loaded with {len(tset)} samples")

    # Initialize cache dictionary to store embeddings
    # Format: {image_path: (label, embedding_tensor)}
    cache = {}

    print(f"🔄 Extracting embeddings from {len(tset)} images...")

    # Process dataset with inference disabled for efficiency
    with torch.inference_mode():
        # Extract features for each image in the dataset
        for fname, (x, y, _) in tqdm(
            zip(tset.image_paths, tset),
            total=len(tset),
            desc=f"Processing {DTS_TO_LOAD}",
            unit="img",
            colour="green"
        ):
            # Normalize file path format (handle Windows/Unix path differences)
            fname = fname.replace("\\", "/")

            # Prepare input tensor: add batch dimension and move to device
            x = x.to(device).unsqueeze(0)

            # Extract features using the DINOv2 model
            f = dino(x)

            # Store the label and embedding in cache
            # f[0] extracts the feature vector from the batch
            cache[fname] = (y, f[0])

    # Save the embeddings cache to disk
    output_path = EMBEDDINGS_FOLDER / f"{DTS_TO_LOAD}.pth"
    print(f"💾 Saving embeddings to: {output_path}")
    torch.save(cache, str(output_path))

    print("✨ Embedding extraction completed successfully!")
    print(f"📁 Output file: {output_path}")
    print(f"📈 Total embeddings extracted: {len(cache)}")
