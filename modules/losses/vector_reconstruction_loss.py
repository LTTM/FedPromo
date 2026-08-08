"""
Module to define the reconstruction loss function.
Classes:
    VectorReconstructionLoss: reconstruction loss function that combines L2, L1 and Cosine losses.
Functions:
    None
Constants:
    None
Exceptions:
    None
Authors: Francesco Barbato
         Matteo Caligiuri
"""

from typing import Optional, Literal
import torch
from torch.nn import Module, MSELoss, L1Loss, CosineSimilarity

__all__ = ["VectorReconstructionLoss"]


class VectorReconstructionLoss(Module):
    """
    Loss function combining L2, L1 and Cosine losses.
    Overides the nn.Module class as required by the PyTorch framework.
    It requires just input1 and input2 tensors and it sets the target tensor to 1.
    """

    def __init__(
        self,
        reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
        dim: Optional[int] = 1,
        lambda_mse: Optional[float] = 1.0,
        lambda_l1: Optional[float] = 1.0,
        lambda_cos: Optional[float] = 1.0,
    ):
        """
        Initializes the VectorReconstructionLoss module.

        Args:
            reduction (Optional[Literal['mean', 'sum', 'none']]):
                The reduction method to apply to the loss.
                Default is 'mean'.
            dim (Optional[int]): The dimension along which to compute the similarity
                for cosine loss.
                Default is 1.
            lambda_mse (Optional[float]): Weighting factor for the MSE loss. Default is 1.
            lambda_l1 (Optional[float]): Weighting factor for the L1 loss. Default is 1.
            lambda_cos (Optional[float]): Weighting factor for the Cosine similarity loss.
                Default is 1.
        """
        super().__init__()

        self.reduction = reduction
        self.dim = dim
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.lambda_cos = lambda_cos
        self.mse = MSELoss(reduction="none")
        self.l1 = L1Loss(reduction="none")
        self.cos = CosineSimilarity(dim=dim)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined loss function.

        Args:
            input1 (torch.Tensor): The first input tensor.
            input2 (torch.Tensor): The second input tensor.

        Returns:
            Any: The computed loss. It can be 'mean', 'sum' or 'none',
                depending on the reduction setting.
        """

        # target tensors are detached to ensure correct gradient computation
        mse = self.mse(input1, input2.detach()).mean(dim=self.dim)
        l1 = self.l1(input1, input2.detach()).mean(dim=self.dim)
        cos = 1.0 - self.cos(
            input1, input2.detach()
        )  # We want to maximize the similarity

        tot_loss = (
            self.lambda_mse * mse + self.lambda_l1 * l1 + self.lambda_cos * cos
        )

        if self.reduction == "mean":
            return tot_loss.mean()
        elif self.reduction == "sum":
            return tot_loss.sum()
        else:
            return tot_loss
