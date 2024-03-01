"""One layer linear neural network model to solve regression problems."""
import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """Linear neural network model used to solve linear regression problems."""

    def __init__(self, number_of_features: int):
        """
        Constructor for LinearRegressionModel.

        :param number_of_features: number of features used to compute a label.
        """
        super().__init__()

        # Randomly initialize as many weights as there are feature and randomly initialize a bias.
        self.weights = nn.Parameter(torch.rand(number_of_features), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        """Forward function for the neural network.

        Pass the input instances through a linear layer: return 'instances*weights + bias'.
        """
        return torch.matmul(instances, self.weights) + self.bias