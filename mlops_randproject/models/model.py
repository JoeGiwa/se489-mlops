import torch.nn as nn
from torch import Tensor


class MyNeuralNet(nn.Module):  # type: ignore[misc]
    """
    Basic feedforward neural network.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()  # ✅ Correctly initializes the base class
        self.l1 = nn.Linear(in_features, 500)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(500, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor with shape [N, in_features].

        Returns:
            Tensor: Output tensor with shape [N, out_features].
        """
        return self.l2(self.relu(self.l1(x)))

    pass
