import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Fruits

device = torch.device("cuda:0")


def visualize_samples(dataset, indices, title=None, count=10):
    """Show images with given indices.
    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        indices (list): List of indices.
        title (str): Title for group of images (default None).
        count (int): Number of images to be shown (default 10).
        
    """
    plt.figure(figsize=(count * 3, 3))
    display_indices = indices[:count]

    if title:
        plt.suptitle("%s %s/%s" % (title, len(display_indices), len(indices)))

    for i, index in enumerate(display_indices):
        x, y, _ = dataset[index]
        plt.subplot(1, count, i + 1)
        plt.title("Label: %s" % y)
        plt.imshow(x)
        plt.grid(False)
        plt.axis('off')


def predicting_fruit(model, folder='/demonstration', visualize=True, count=10,
                     title=None, title_list=None):
    """Demonstrate results of the model.
       
       Show image with predicted class name.
       
       Args:
           model (torch.nn): Model of neuron network.
           folder (str): Folder with images (default /demonstration).
           visualize (bool): Should images be shown or not (default True).
           count (int): Number of images to be shown (default 10).
           title (str): Title for group of images (default None).
           title_list (dict): Dict with class names and indices (default None).
           
    """
    # Dataset for image showing.
    orig_dataset = Fruits(folder, production=True)
    # Dataset for predicting of class labels.
    load_dataset = Fruits(folder,
                          transform=transforms.Compose([
                              transforms.Resize((224, 224)),  # Size for network.
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                          ]), production=True  # True because there is no information about ground truth class label.
                          )
    load_loader = torch.utils.data.DataLoader(load_dataset, batch_size=1)

    model.eval()
    predictions = []  # List for predicted class labels.

    for inputs, _, _ in load_loader:
        inputs_gpu = inputs.to(device)  # Sends to GPU.
        prediction = model(inputs_gpu)
        _, prediction_semi = torch.max(prediction, 1)  # Gets predicted class labels.
        predictions += prediction_semi.tolist()

    if visualize:  # Shows images if argument is True.
        plt.figure(figsize=(count * 3, 3))
        display_indices = list(range(len(orig_dataset)))
        if title:
            plt.suptitle("%s %s/%s" % (title, len(display_indices), len(orig_dataset)))

        for i, index in enumerate(display_indices):
            x, _, _ = orig_dataset[index]
            y = title_list[predictions[index]]  # Images will be shown with predicted class names.
            plt.subplot(1, count, i + 1)
            plt.title("Label: %s" % y)
            plt.imshow(x)
            plt.grid(False)
            plt.axis('off')

    return [title_list[i] for i in predictions]


def image_show(inp, title=None):
    """Function for showing image from tensor.
    Args:
        inp (torch.tensor): Tensor.
        title (str): Title for group of images (default None).
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)
