"""
Script containing all classifiers used in this work.

Classes:
    SimpleClassifier: A simple neural network classifier for multi-class classification tasks.

Functions:
    None

Constants:
    None

Exceptions:
    ValueError: Raised when an invalid dropout rate is provided.

Authors: Francesco Barbato
         Matteo Caligiuri
"""

from typing import Optional


import torch
from torch import nn


# Define the default import behavior for the module.
__all__ = ["SimpleClassifier"]


class SimpleClassifier(nn.Module):
    """
    A simple neural network classifier for multi-class classification tasks.

    Args:
        features (int): The number of input features.
        num_classes (int): The number of output classes.
        num_layers (int): The number of hidden layers in the network.
            Defaults to 1.
        dropouts (float): The dropout rate to apply to the hidden layers.
            Defaults to None.

    Attributes:
        fcn (nn.Sequential): The fully connected network of the classifier.

    Methods:
        forward(self, x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the model using the input features
                and returns the class logits.

    Exceptions:
        ValueError: Raised when an invalid dropout rate is provided.
    """

    def __init__(
        self,
        features: int,
        num_classes: int,
        num_layers: Optional[int] = 1,
        dropouts: Optional[float] = None,
    ) -> None:

        super().__init__()

        # Define the hidden layers of the network.
        layers = []
        for layer_index in range(num_layers):
            # Define the input and output features for the current layer.
            if layer_index == 0:
                in_feats = features
            else:
                prev_layer = layers[-1]
                if isinstance(prev_layer, nn.Dropout):
                    in_feats = layers[-2].out_features
                if isinstance(prev_layer, nn.ReLU):
                    in_feats = layers[-2].out_features
                else:
                    in_feats = prev_layer.out_features

            if layer_index == num_layers - 1:
                out_feats = num_classes
            elif in_feats // 2 > num_classes:
                out_feats = in_feats // 2
            else:
                out_feats = in_feats

            if layer_index < num_layers - 1:
                layers.append(nn.Linear(in_feats, out_feats))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(in_feats, out_feats, bias=False))

            # Add the dropout layer if specified.
            if dropouts is not None:
                # Check that the dropout rate is valid.
                if not 0.0 <= dropouts < 1.0:
                    raise ValueError(
                        f"Invalid dropout rate: {dropouts}. Must be in the range [0, 1)."
                    )

                # Add the dropout layer to the network.
                layers.append(nn.Dropout(dropouts))

        # Define the fully connected layer of the network.
        self.fcn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model using the input features
            and returns the class logits.

        Args:
            x (torch.Tensor): The input features to the model.

        Returns:
            torch.Tensor: The class logits for the input features.
        """

        return self.fcn(x)
