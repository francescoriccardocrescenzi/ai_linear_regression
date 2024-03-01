"""Synthetic linear data generation."""
import torch


class SyntheticLinearDataset(torch.utils.data.Dataset):
    """Create synthetic linear data polluted by noise according to a normal distribution."""

    def __init__(self, weights: torch.Tensor, bias: torch.Tensor, noise: float, number_of_instances: int):
        """
        Constructor for SyntheticLinearDataset.

        Store weights, bias, and noise as attributes. Generate the linear data and pollute it with noise.

        :param weights: True weights used to generate linear data. 'weights.size(0)' will be taken as the number of
            features in the generated data.
        :param bias: True bias used to generate linear data.
        :param noise: Standard deviation of the noise.
        :param number_of_instances: number of instances that are to be generated.
        """
        # Store the parameters.
        self.weights = weights.unsqueeze(-1)
        self.bias = bias
        self.noise = noise

        # Generate the linear data and pollute it with noise.
        self.instances = torch.rand(number_of_instances, self.weights.size(0))
        self.labels = torch.matmul(self.instances, self.weights) + self.bias
        self.labels += (torch.randn(number_of_instances, self.weights.size(0))*self.noise)

    def __getitem__(self, item):
        """Overload indexing to allow DataLoader to retrieve the data."""
        return self.instances[item], self.labels[item]

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.instances)