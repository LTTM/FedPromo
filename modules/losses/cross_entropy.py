"""
Module defining a non-categorical cross-entropy loss.

Classes:
    CrossEntropyLoss: Non-categorical cross-entropy loss.

Functions:
    None

Constants:
    None

Exceptions:
    None

Author: Francesco Barbato
        Matteo Caligiuri
"""

from typing import Literal, Optional

import torch
import torch.nn as nn


# Define what to export
__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(nn.Module):
    """
    Non-categorical cross-entropy loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            Default: 'mean'
    """

    def __init__(
        self,
        reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
    ):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(
        self, input: torch.Tensor, target: torch.Tensor  # pylint: disable=redefined-builtin
    ) -> torch.Tensor:
        """
        Compute the non-categorical cross-entropy loss.

        Args:
            input (torch.Tensor): The input tensor (logits).
            target (torch.Tensor): The target tensor (logits).

        Returns:
            torch.Tensor: The computed loss.
        """

        gt_log_prob = torch.nn.functional.log_softmax(target.detach(), dim=1)
        prob = torch.nn.functional.softmax(input, dim=1)

        ent = -torch.sum(prob * gt_log_prob, dim=1)

        if self.reduction == "mean":
            return torch.mean(ent)
        elif self.reduction == "sum":
            return torch.sum(ent)
        elif self.reduction == "none":
            return ent
