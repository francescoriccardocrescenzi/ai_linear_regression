"""Visualization tools to compare actual data with neural network predictions."""
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


def plot_predictions(model: nn.Module, dataloader: DataLoader, title: str = None):
    """Take a random batch of data and plot the model-predicted labels against the actual labels.

    :param model: neural network.
    :param dataloader: dataloader used to load the random batch.
    :param title: title of the plot.
    :returns: None
    """
    instances, labels = next(iter(dataloader))
    with torch.inference_mode():
        predictions = model(instances)

    plt.figure()
    plt.title(title)
    plt.scatter(instances, labels, c='b', s=4, label="Actual data")
    plt.scatter(instances, predictions, c='r', s=4, label="Model predictions")
    plt.legend()
    plt.show()