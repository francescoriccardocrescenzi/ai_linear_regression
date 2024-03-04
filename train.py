"""Trainer object to train linear regression models."""
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """Train an instance of linear regression model.

    Every instance is initialized with a given training dataloader, testing dataloader, loss function and optimizer.
    Use these objects to execute an appropriate training loop.
    """
    def __init__(self, model: nn.Module, training_dataloader: DataLoader, testing_dataloader: DataLoader,
                 loss_function, optimizer):
        """Constructor for Trainer.

        :param model: model to be trained.
        :param training_dataloader: dataloader for the training dataset.
        :param testing_dataloader: dataloader for the testing dataset.
        :param loss_function: loss function.
        :param optimizer: optimizer.
        """
        self.model = model
        self.training_dataloader = training_dataloader
        self.testing_dataloader = testing_dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train_one_epoch(self):
        """Train the model for a single epoch."""
        for instances, labels in self.training_dataloader:
            self.model.train()
            predictions = self.model(instances).unsqueeze(-1)
            loss = self.loss_function(predictions, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, epochs: int):
        """Execute the training loop for the given number of epochs."""
        for epoch in range(epochs):
            self.train_one_epoch()


