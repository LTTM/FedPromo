"""
Module to handle the CLIP model from Hugging Face.

Classes:
    CLIP: Wrapper to initialize and handle the CLIP model.

Functions:
    instantiate_clip: Instantiate the CLIP model using the OPenAI implementation.
    create_labels: Create the labels for the model.
    create_labels_embeddings: Create the label embeddings.
    get_class_scores: Get the class scores.
    labels: Add the labels property.
    device: Add the device property.
    tokenize: Add the tokenize property.

Constants:
    None

Escceptions:
    ValueError: Raised when the provided pretrained model is not available.
    ValueError: Raised when the labels are not defined.
    ValueError: Raised when either image or image_features must be defined.

Author: Matteo Caligiuri
"""

from typing import List, Dict, Union, Any, Optional, Callable
from pathlib import Path
import warnings

import torch
import clip
from clip.model import CLIP

from modules.common import (
    register_method,
    register_new_class_elements,
    register_property,
)


# Set the target class for the decorators
_target_class = CLIP  # pylint: disable=invalid-name


# Ignore the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


__all__ = ["instantiate_clip"]


def instantiate_clip(
    pretrained_model: str,
    data_folder: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cpu"),  # pylint: disable=redefined-outer-name
    **kwargs,  # pylint: disable=unused-argument
) -> CLIP:
    """
    Instantiate the CLIP model and processor from Hugging Face.

    Args:
        pretrained_model (str): The pretrained model.
        data_folder (Optional[Union[Path, str]], optional):
            The data folder where the model will be downlaoded.
            Defaults to None.
        device (torch.device, optional): The device to use.
            Defaults to torch.device("cpu").
        **kwargs: Additional arguments to pass to the model.
    Returns:
        CLIP: The CLIP model.
    """

    # Check if the provided pretrained model is available
    if pretrained_model not in clip.available_models():
        raise ValueError(
            f"The provided pretrained model {pretrained_model} is not available."
        )

    # Convert the data_folder to a Path object
    # and create it if it does not exist
    data_folder = Path(data_folder) if isinstance(data_folder, str) else data_folder
    data_folder = data_folder / "clip_models"
    data_folder.mkdir(parents=True, exist_ok=True)

    # Load the model
    model = clip.load(name=pretrained_model, device=device, download_root=data_folder)[
        0
    ]

    # Register the added methods and properties
    register_new_class_elements(model)

    # Set the device
    model.device = device

    return model


@register_method
def create_labels(
    self,
    labels: Union[List[str], Dict[Any, str]],  # pylint: disable=redefined-outer-name
    split_token: Optional[str] = ", ",
    prepend_phrase: Optional[str] = "a photo of a ",
) -> None:
    """
    Create the labels for the model.
    The labels are the first part of the string before the split token (default is ", ").
    To this label is preppended the phrase "a photo of a ".

    Args:
        labels (Union[List[str], Dict[Any, str]]): The labels.
        split_token (Optional[str], optional): The token to split the labels.
            Defaults to ", ".
        prepend_phrase (Optional[str], optional): The phrase to prepend to the labels.
            Defaults to "a photo of a ".
    Returns:
        None
    """

    if isinstance(labels, dict):
        labels = list(labels.values())

    # Check that the prepend_phrase end with a space
    if not prepend_phrase.endswith(" "):
        prepend_phrase += " "

    self.labels = [prepend_phrase + value.split(split_token)[0] for value in labels]


@register_method
@torch.no_grad()
def create_labels_embeddings(
    self,
    normalize: Optional[bool] = True,
    labels: Optional[  # pylint: disable=redefined-outer-name
        Union[List[str], Dict[Any, str]]
    ] = None,
    split_token: Optional[str] = ", ",
    prepend_phrase: Optional[str] = "a photo of a ",
) -> None:
    """
    Create the label embeddings.

    Args:
        normalize (Optional[bool], optional): Normalize the embeddings. Defaults to True.
        labels (Optional[Union[List[str], Dict[Any, str]]], optional): The labels.
            Defaults to None.
        split_token (Optional[str], optional): The token to split the labels. Defaults to ", ".
        prepend_phrase (Optional[str], optional): The phrase to prepend to the labels.
            Defaults to "a photo of a ".
    Returns:
        None
    """

    # Check if the labels are defined or not, if possible create them
    if self.labels is None and labels is None:
        raise ValueError("The labels are not defined.")
    elif labels is not None:
        self.create_labels(
            labels=labels, split_token=split_token, prepend_phrase=prepend_phrase
        )

    # Create label tokens
    label_tokens = self.tokenize(self.labels).to(self.device)

    # Encode tokens to sentence embeddings
    label_feats = self.encode_text(label_tokens)

    # Normalization
    if normalize:
        label_feats = label_feats / torch.norm(label_feats, dim=0)

    self.label_feats = label_feats


@register_method
@torch.no_grad()
def get_class_scores(
    self,
    image: Optional[torch.Tensor] = None,
    image_feats: Optional[torch.Tensor] = None,
    normalize: Optional[bool] = True,
    compute_pred: Optional[bool] = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Get the class scores

    Args:
        image (torch.Tensor): The image/s.
        image_feats (torch.Tensor): The image features (normalized).
        normalize (Optional[bool]): Normalize the image features. Defaults to True.
        compute_pred (Optional[bool]): Compute the prediction. Defaults to False.
    Returns:
        Union[torch.Tensor, Dict[str, torch.Tensor]]: The class final prediction
            (if compute_pred=True) or a dictionary with the logits and the features.
    """

    # Check if image or image_features are defined
    # and act accordingly to get the image features
    if image is not None:
        image_feats = self.encode_image(image=image)
    elif image is None and image_feats is None:
        raise ValueError("Either image or image_features must be defined.")

    # Get the clean features
    clean_feats = image_feats

    # Normalization
    if normalize:
        image_feats = image_feats / torch.norm(image_feats, dim=0)

    # Compute the scores
    scores = torch.matmul(image_feats, torch.transpose(self.label_feats, 0, 1))

    # Compute the prediction if required
    if compute_pred:
        return torch.argmax(scores, dim=1)

    # Return the logits and the features
    return {"logits": scores, "feats": clean_feats}


@register_method
def get_feats_shape(self) -> int:
    """
    Get the features shape.

    Args:
        None

    Returns:
        int: The features shape.
    """

    return self.ln_final.normalized_shape[0]


@register_property
def labels() -> None:
    """
    Add the labels property.
    """

    return None


@register_property
def device() -> None:
    """
    Add the device property.
    """

    return None


@register_property
def tokenize() -> Callable:
    """
    Add the tokenize property.
    """

    return clip.tokenize
