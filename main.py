"""Create a one layer linear neural network and train it to solve a linear regression problem with synthetic data."""
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from data import SyntheticLinearDataset
from model import LinearRegressionModel
from visualization import plot_predictions


def training_loop(model: nn.Module, dataloader: DataLoader, number_of_epochs: int, loss_function, optimizer):
    """raining loop for LinearRegressionModel.

    Train the model using backpropagation and minibatch stochastic gradient descent (MSGD).

    :param model: model to be trained.
    :param dataloader: dataloader for the training data.
    :param number_of_epochs: number of training epochs.
    :param loss_function: loss function.
    :param optimizer: optimizer.
    :returns: None
    """
    for epoch in range(number_of_epochs):
        # Apply MSGD
        for instances, labels in dataloader:
            model.train()
            predictions = model(instances).unsqueeze(-1)
            loss = loss_function(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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
batch_size = 100
num_epochs = 100

# Create optimizer and loss function.
mse_loss_function = nn.MSELoss()
sgd_optimizer = torch.optim.SGD(params=regression_model.parameters(), lr=learning_rate)

# Create data loaders.
training_dataloader = DataLoader(training_dataset, batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size, shuffle=True)

# Visualize predictions before training.
plot_predictions(regression_model, testing_dataloader, title="Before training")

# Execute training loop.
training_loop(regression_model, training_dataloader, num_epochs, mse_loss_function, sgd_optimizer)

# Visualize predictions after training.
plot_predictions(regression_model, testing_dataloader, title="After training")



