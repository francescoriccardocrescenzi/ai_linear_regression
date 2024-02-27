"""Linear neural network for solving a linear regression problem.

The neural network is powered using PyTorch. The results are plotted using matplotlib.
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


def plot_predictions(model: nn.Module, dataloader: DataLoader):
    """Take a random batch of data and plot the model-predicted labels against the actual labels.

    :param model: neural network.
    :dataloader: dataloarder used to load the random batch.
    """
    instances, labels = next(iter(dataloader))
    with torch.inference_mode():
        predictions = model(instances)

    plt.figure()
    plt.scatter(instances, labels, c='b', s=4)
    plt.scatter(instances, predictions, c='r', s=4)
    plt.show()


class SyntheticLinearDataset(Dataset):
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


# Create and split dataset. Use an 80% vs 20% split for training data vs testing data.
number_of_features = 1
number_of_instances = 10000
weights = torch.rand(number_of_features)
bias = torch.rand(1)
noise = 0.01

dataset = SyntheticLinearDataset(weights, bias, noise, number_of_instances)
training_length = int(number_of_instances*0.8)
testing_length = number_of_instances - training_length
training_dataset, testing_dataset = random_split(dataset, [training_length, testing_length])

# Create model.
regression_model = LinearRegressionModel(number_of_features)

# Create hyperparameters.
learning_rate = 0.1
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=regression_model.parameters(), lr=learning_rate)
batch_size = 100

# Create data loaders.
training_dataloader = DataLoader(training_dataset, batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size, shuffle=True)

# Visualize predictions before training.
plot_predictions(regression_model, testing_dataloader)


def training_loop(model: nn.Module, dataloader: DataLoader):
    """Training loop for LinearRegressionModel."""
    num_epochs = 100
    for epoch in range(num_epochs):
        for instances, labels in dataloader:
            model.train()
            predictions = model(instances).unsqueeze(-1)
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Execute training loop.
training_loop(regression_model, training_dataloader)

# Visualize predictions after training.
plot_predictions(regression_model, testing_dataloader)



