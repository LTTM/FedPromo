"""
Module to define the cosine similarity loss function.

Classes:
    CosineSimilarityLoss: Cosine similarity loss function.

Functions:
    None

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
"""

import torch
from torch.nn import CosineEmbeddingLoss


__all__ = ["CosineSimilarityLoss"]


class CosineSimilarityLoss(CosineEmbeddingLoss):
    """
    Cosine similarity loss function.
    Overide the CosineEmbeddingLoss class to provide a more standard interface.
    It requires just input1 and input2 tensors and it sets the target tensor to 1.
    """

    def __call__(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity loss.

        Args:
            input1 (torch.Tensor): The first input tensor.
            input2 (torch.Tensor): The second input tensor.

        Returns:
            Any: The computed loss.
        """

        return super().__call__(
            input1, input2, torch.ones(input1.size(0)).to(input1.device)
        )
