import os

from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset
from collections import namedtuple

fruit_list = {0: 'Apple', 1: 'Banana', 2: 'Carambola', 3: 'Guava',
              4: 'Kiwi', 5: 'Mango', 6: 'Muskmelon', 7: 'Orange', 8: 'Peach',
              9: 'Pear', 10: 'Persimmon', 11: 'Pitaya', 12: 'Plum',
              13: 'Pomegranate', 14: 'Tomato', 15: 'Not a fruit'}


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
        self.list_of_paths = []
        for root, dirs, files in os.walk(folder):
            if len(dirs) != 0:
                continue
            for name in files:
                self.list_of_paths.append(os.path.join(root, name))
        self.production = production

    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.list_of_paths)

    def __getitem__(self, index):
        """Return image, class label and file path.
        Args:
            index (int): Indice of the element in the dataset.
        """
        img_id = None
        img_id = self.list_of_paths[index]
        img = Image.open(img_id).convert('RGB')
        if self.production:  # True when we can't get ground truth label (for new images).
            y = 'Class to be predicted'
        else:  # For training mode ground truth can be get from folder name.
            inv_fruit_list = dict(zip(fruit_list.values(), fruit_list.keys()))
            class_title = str.split(img_id, sep='/')[1]  # Get class name.
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
        """Return list of indices."""
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        """Return number of indices in the list."""
        return len(self.indices)


fruit_nutrition = namedtuple(
    "Nutritional value", ['Edible', 'Energy', 'Carbohydrates', 'Sugars', 'Dietary fiber', 'Fat', 'Protein']
)

fruit_description = {
    0: fruit_nutrition('Yes', '218 kJ (52 kcal)', '13.81 g', '10.39 g', '2.4 g', '0.17 g', '0.26 g'),
    1: fruit_nutrition('Yes', '371 kJ (89 kcal)', '22.84 g', '12.23 g', '2.6 g', '0.33 g', '1.09 g'),
    2: fruit_nutrition('Yes', '128 kJ (31 kcal)', '6.73 g', '3.98 g', '2.8 g', '0.33 g', '1.04 g'),
    3: fruit_nutrition('Yes', '285 kJ (68 kcal)', '14.32 g', '8.92 g', '5.4 g', '0.95 g', '2.55 g'),
    4: fruit_nutrition('Yes', '255 kJ (61 kcal)', '14.66 g', '8.99 g', '3 g', '0.52 g', '1.14 g'),
    5: fruit_nutrition('Yes', '250 kJ (60 kcal)', '15 g', '13.7 g', '1.6 g', '0.38 g', '0.82 g'),
    6: fruit_nutrition('Yes', '250 kJ (60 kcal)', '14.4 g', '13.9 g', '1.6 g', '0.9 g', '0.9 g'),
    7: fruit_nutrition('Yes', '197 kJ (47 kcal)', '11.75 g', '9.35 g', '2.4 g', '0.12 g', '0.94 g'),
    8: fruit_nutrition('Yes', '274 kJ (65 kcal)', '16 g', '14.10 g', '2.5 g', '0.42 g', '1.53 g'),
    9: fruit_nutrition('Yes', '239 kJ (57 kcal)', '15.23 g', '9.75 g', '3.1 g', '0.14 g', '0.36 g'),
    10: fruit_nutrition('Yes', '293 kJ (70 kcal)', '18.59 g', '12.53 g', '3.6 g', '0.19 g', '0.58 g'),
    11: fruit_nutrition('Yes', '250 kJ (60 kcal)', '82.14 g', '82.14 g', '1.8 g', '0.3 g', '3.57 g'),
    12: fruit_nutrition('Yes', '192 kJ (46 kcal)', '11.42 g', '9.92 g', '1.4 g', '0.28 g', '0.7 g'),
    13: fruit_nutrition('Yes', '346 kJ (83 kcal)', '18.7 g', '13.67 g', '4 g', '1.17 g', '1.67 g'),
    14: fruit_nutrition('Yes', '74 kJ (18 kcal)', '3.9 g', '2.6 g', '1.2 g', '0.2 g', '0.9 g'),
    15: 'It can be either unknown type of fruit or not a fruit.'
}
