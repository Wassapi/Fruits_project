import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def visualize_samples(dataset, indices, title=None, count=10):
    plt.figure(figsize=(count*3,3))
    display_indices = indices[:count]
    if title:
        plt.suptitle("%s %s/%s" % (title, len(display_indices), len(indices)))        
    for i, index in enumerate(display_indices):    
        x, y, _ = dataset[index]
        plt.subplot(1,count,i+1)
        plt.title("Label: %s" % y)
        plt.imshow(x)
        plt.grid(False)
        plt.axis('off') 

def predicting_fruit(model, folder = '/demostration', visualize = True, count = 10, 
                     title = None, title_list = None):
    orig_dataset = Fruits(load_folder, production = True)
    load_dataset = Fruits(load_folder, 
                          transform=transforms.Compose([
                          transforms.Resize((224, 224)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                          ]), production = True
                      )
    load_loader = torch.utils.data.DataLoader(load_dataset, batch_size=1)
    model.eval()
    predictions = []
    for k, (inputs, gt, id) in enumerate(load_loader):
        inputs_gpu = inputs.to(device)
        prediction = model(inputs_gpu)
        _, prediction_semi = torch.max(prediction, 1)
        predictions += prediction_semi.tolist()
    if visualize == True:
        plt.figure(figsize=(count*3,3))
        display_indices = list(range(len(orig_dataset)))
        if title:
            plt.suptitle("%s %s/%s" % (title, len(display_indices), len(orig_dataset)))        
        for i, index in enumerate(display_indices):    
            x, _, _ = orig_dataset[index]
            y = title_list[predictions[index]]
            plt.subplot(1,count,i+1)
            plt.title("Label: %s" % y)
            plt.imshow(x)
            plt.grid(False)
            plt.axis('off')  
    return [Fruit_list[i] for i in predictions]

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)