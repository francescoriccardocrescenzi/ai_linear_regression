"""Create a one layer linear neural network and train it to solve a linear regression problem with synthetic data."""
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from data import SyntheticLinearDataset
from model import LinearRegressionModel
from visualization import plot_predictions
from train import Trainer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Create HYPERPARAMETERS.
learning_rate = 0.1
weight_decay = 0.001
batch_size = 100
num_epochs = 100


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
model = LinearRegressionModel(number_of_features)

# Create optimizer and loss function.
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create data loaders.
training_dataloader = DataLoader(training_dataset, batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size, shuffle=True)

# Visualize predictions before training.
plot_predictions(model, testing_dataloader, title="Before training")

# Create summary writer to log losses to TensorBoard.
# Use a timestamp to differentiate between runs in the TensorBoard logs.
timestamp = int(datetime.now().timestamp())
with SummaryWriter(log_dir=f"logs/{timestamp}") as summary_writer:

    # Create Trainer object and execute training loop.
    trainer = Trainer(model, training_dataloader, testing_dataloader, loss_function, optimizer, summary_writer)
    trainer.train(epochs=num_epochs)

# Visualize predictions after training.
plot_predictions(model, testing_dataloader, title="After training")
