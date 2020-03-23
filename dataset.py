import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler, Sampler
import os
from PIL import Image

fruit_list = {0:'Apple', 1:'Banana', 2:'Carambola', 3:'Guava',
              4:'Kiwi', 5:'Mango', 6:'Muskmelon', 7:'Orange', 8:'Peach',
              9:'Pear', 10:'Persimmon', 11:'Pitaya', 12:'Plumу',
              13:'Pomegranet', 14:'Tomato', 15:'Not a fruit'}

class Fruits(Dataset):
    """Create dataset from all images in the folder and subfolders"""
    def __init__(self, folder, transform=None, production=False):
        """Create list of all paths of the files.
        Args:
            folder (str): Folder name with all images.
            transform (torch.transforms): Augmentation for images (default None).
            production (bool): False for training mode True for predicting class for unallocated images (default False).
        """
        self.transform = transform
        self.folder = folder
        self.listname = []
        oswalk = os.walk(folder)
        for root,dirs,files in oswalk:
            if len(dirs) !=0:
                continue
            for name in files:
                self.listname.append(os.path.join(root, name))
        self.production = production
        
    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.listname)
    
    def __getitem__(self, index):
        """Return image, class label and file path.
        Args:
            index (int): Indice of the element in the dataset.
        """
        img_id = None
        img_id = self.listname[index]
        img = Image.open(img_id).convert('RGB')
        if self.production == True: # True when we can't get ground truth label (for new images).
            y = 'Class to be predicted'
        else: # For training mode ground truth can be get from folder name (every class in data is in the seperate folder).
            inv_fruit_list = dict(zip(fruit_list.values(), fruit_list.keys()))
            class_title = str.split(img_id, sep= '/')[1] # Get class name as name of the folder where image is located (only during training). 
            if class_title == 'Other':
                y = 15
            else:
                # Get class label from class name.
                try:
                    y = inv_fruit_list[class_title]
                except KeyError:
                    raise KeyError(
                        'Folder with unknown class:', 
                        class_title)

        if self.transform:
            img = self.transform(img)
        return img, y, img_id

    
class SubsetSampler(Sampler):
    """Class for Sampler, which return indices in order."""

    def __init__(self, indices):
        """Create list of indices.
        Args:
            indices (list): List of indices.
        """
        self.indices = indices

    def __iter__(self):
        """Iteration for list."""
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        """Return number of indices in the list."""
        return len(self.indices)