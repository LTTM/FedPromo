"""
Module that implement the instantiation of the CLIP model with the fine-tuning
for INaturalist dataset. The fine-tuning was performed as in the following paper
https://arxiv.org/abs/2304.01457

Classes:
    CLIPModified: Override of the CLIP class just for renaming it
    VisionTransformerModified: Override of the VisionTransformer class of CLIP

Functions:
    instantiate_clip_fine: Instantiate the CLIP model with fine-tuning.
    get_class_scores: Get the class scores.
    replace_vit: Replace the Vision Transformer with the custom one.
    labels: Add the labels property.
    device: Add the device property.
    decoder: Add the decoder property.
    classifier: Add the classifier property.

Constants:
    _target_class: The target class for the decorators.

Exceptions:
    ValueError: Raised when either image or image_features must be defined.

Author: Matteo Caligiuri
"""

from typing import Union, Optional, Dict
from pathlib import Path

import torch
from torch import nn
import clip
from clip.model import CLIP, VisionTransformer
from timm.models.vision_transformer import Block

from modules.common import (
    register_method,
    register_new_class_elements,
    register_property,
)


__all__ = ["instantiate_clip_fine"]


class CLIPModified(CLIP):
    """
    Override of the CLIP class just for renaming it

    Args:
        embed_dim (int): The dimension of the embeddings.
        image_resolution (int): The resolution of the image.
        vision_layers (Union[Tuple[int, int, int, int], int]): The number of layers
            of the vision transformer.
        vision_width (int): The width of the vision transformer.
        vision_patch_size (int): The patch size of the vision transformer.
        context_length (int): The context length of the transformer.
        vocab_size (int): The size of the vocabulary.
        transformer_width (int): The width of the transformer.
        transformer_heads (int): The number of heads of the transformer.
        transformer_layers (int): The number of layers of the transformer.
    """


class VisionTransformerModified(VisionTransformer):
    """
    Override of the VisionTransformer class of CLIP.
    It replaces  just the forward function to allign it with the new decoder and classifier.
    """

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


# Set the target class for the decorators
_target_class = CLIPModified  # pylint: disable=invalid-name


def instantiate_clip_fine(
    checkpoint_path: str,
    data_folder: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cpu"),  # pylint: disable=redefined-outer-name
    **kwargs,  # pylint: disable=unused-argument
) -> CLIPModified:
    """
    Instantiate the CLIP model with the fine-tuning for INaturalist dataset.

    Args:
        checkpoint_path (str): The path to the checkpoint.
        data_folder (Optional[Union[Path, str]], optional):
            The data folder where the model will be downlaoded.
            Defaults to None.
        device (torch.device, optional): The device to use.
            Defaults to torch.device("cpu").
        **kwargs: Additional arguments to pass to the model

    Returns:
        CLIPModified: The CLIP model.
    """

    # Convert the data_folder to a Path object
    # and create it if it does not exist
    data_folder = Path(data_folder) if isinstance(data_folder, str) else data_folder
    data_folder = data_folder / "clip_models"
    data_folder.mkdir(parents=True, exist_ok=True)

    # Load the model
    model = clip.load(name="ViT-L/14", device=device, download_root=data_folder)[0]
    model.__class__ = CLIPModified

    # Register the added methods and properties
    register_new_class_elements(model)

    # Chance vision transformer
    model.replace_vit()

    # Set the device
    model.device = device

    # Load the fine-tuned checkpoint
    _load_ckpt(model=model, ckpt_path=Path(checkpoint_path))

    return model


@register_method
@torch.no_grad()
def get_class_scores(
    self,
    image: Optional[torch.Tensor] = None,
    image_feats: Optional[torch.Tensor] = None,
    compute_pred: Optional[bool] = False,
    **kwargs,  # pylint: disable=unused-argument
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Get the class scores

    Args:
        image (torch.Tensor): The image/s.
        image_feats (torch.Tensor): The image features (normalized).
        compute_pred (Optional[bool]): Compute the prediction. Defaults to False.
        **kwargs: Additional arguments.
    Returns:
        Union[torch.Tensor, Dict[str, torch.Tensor]]: The class final prediction
            (if compute_pred=True) or a dictionary with the logits and the features.
    """

    # Check if image or image_features are defined
    # and act accordingly to get the image features
    if image is not None:
        image_feats = self.encode_image(image=image).to(dtype=torch.float32)
    elif image is None and image_feats is None:
        raise ValueError("Either image or image_features must be defined.")

    # Run the feature through the decoder
    vision_outputs = self.decoder(image_feats)
    pooled_output = vision_outputs[:, 0, :]

    # Run the pooled output through the classifier
    scores = self.classifier(pooled_output)

    # Compute the prediction if required
    if compute_pred:
        return torch.argmax(scores, dim=1)

    # Return the logits and the features
    return {
        "logits": scores,
        "feats": pooled_output,
        "image_feats": image_feats[:, 0, :],
    }


@register_method
def replace_vit(self) -> None:
    """
    Replace the Vision Transformer with the custom one.

    Args:
        self (CLIPModified): The CLIP model.

    Returns:
        None
    """

    self.visual.__class__ = VisionTransformerModified


@register_method
def get_feats_shape(self) -> int:
    """
    Get the features shape.

    Args:
        None

    Returns:
        int: The features shape.
    """

    return self.classifier.in_features


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
def decoder() -> nn.Module:
    """
    Add the decoder property.
    """

    return nn.Sequential(*[Block(1024, 4, 0.5, qkv_bias=True) for i in range(3)]).to(
        "cuda"
    )


@register_property
def classifier() -> nn.Module:
    """
    Add the classifier property.
    """

    return nn.Linear(1024, 8142).to("cuda")


def _load_ckpt(model: CLIPModified, ckpt_path: Path) -> None:
    """
    Load the fine-tuned checkpoint.

    Args:
        model (CLIPModified): The CLIP model.
        ckpt_path (Path): The checkpoint path.
    """

    # Extract the standard CLIP state
    clip_state = model.state_dict()

    # Load the checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location=model.device, weights_only=False)[
            "model"
        ]

        # Remove all the keys that are not related to the decoder and the classifier
        ckpt = {
            k: v for k, v in ckpt.items() if "decoder_blocks" in k or "classifier" in k
        }

        # Load the decoder and classifier data to the server model
        ckpt = {
            k.replace("module._orig_mod.decoder_blocks", "decoder"): v
            for k, v in ckpt.items()
        }
        ckpt = {
            k.replace("module.backbone._orig_mod.decoder_blocks", "decoder"): v
            for k, v in ckpt.items()
        }
        ckpt = {
            k.replace("module._orig_mod.classifier", "classifier"): v
            for k, v in ckpt.items()
        }
        ckpt = {
            k.replace("module.backbone._orig_mod.classifier", "classifier"): v
            for k, v in ckpt.items()
        }

        # Save the cleaned state
        torch.save(ckpt, ckpt_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.") from e
    except KeyError:
        ckpt = torch.load(ckpt_path, map_location=model.device, weights_only=False)

    # Update the state
    clip_state.update(ckpt)
    model.load_state_dict(clip_state)
