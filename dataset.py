import os

import random
from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset

fruit_list = {0: 'Apple', 1: 'Banana', 2: 'Carambola', 3: 'Guava',
              4: 'Kiwi', 5: 'Mango', 6: 'Muskmelon', 7: 'Orange', 8: 'Peach',
              9: 'Pear', 10: 'Persimmon', 11: 'Pitaya', 12: 'Plum',
              13: 'Pomegranate', 14: 'Tomato', 15: 'It can be either unknown type of fruit or not a fruit.'}


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


def message_with_nutrition(index):
    """Function return message with phrase and nutrition
    Args:
        index (int): label of the class
    """
    fruit_nutrition_list = ['Carbohydrates', 'Sugars', 'Dietary fiber', 'Fat', 'Protein']
    fruit_description = [
        ['Edible', 218, 52, 13.81, 10.39, 2.4, 0.17, 0.26],
        ['Edible', 371, 89, 22.84, 12.23, 2.6, 0.33, 1.09],
        ['Edible', 128, 31, 6.73, 3.98, 2.8, 0.33, 1.04],
        ['Edible', 285, 68, 14.32, 8.92, 5.4, 0.95, 2.55],
        ['Edible', 255, 61, 14.66, 8.99, 3, 0.52, 1.14],
        ['Edible', 250, 60, 15, 13.7, 1.6, 0.38, 0.82],
        ['Edible', 250, 60, 14.4, 13.9, 1.6, 0.9, 0.9, ],
        ['Edible', 197, 47, 11.75, 9.35, 2.4, 0.12, 0.94],
        ['Edible', 274, 65, 16, 14.10, 2.5, 0.42, 1.53],
        ['Edible', 239, 57, 15.23, 9.75, 3.1, 0.14, 0.36],
        ['Edible', 293, 70, 18.59, 12.53, 3.6, 0.19, 0.58],
        ['Edible', 250, 60, 82.14, 82.14, 1.8, 0.3, 3.57],
        ['Edible', 192, 46, 11.42, 9.92, 1.4, 0.28, 0.7],
        ['Edible', 346, 83, 18.7, 13.67, 4, 1.17, 1.67],
        ['Edible', 74, 18, 3.9, 2.6, 1.2, 0.2, 0.9],
        'I hope it is edible ^-^'
    ]
    phrases = [
        'Say ‘hello’ to my little friend ',
        'Heeeeere’s Johnny with ',
        'I know you are , but what am I?',
        'Surely, you can’t be serious.” – “I am serious, and don’t call me ”',
        'Dammit, man, I’m a DOCTOR, not a – ',
        'WHAT IS YOUR MAJOR MALFUNCTION ',
        'There can be only one ',
        'THIS IS ',
        'Houston, we have a ',
        'Get away from her, you ',
        'Take your stinking paws off me, you damn dirty ',
        'I’m too old for this ',
        'Why so ',
        'I feel the need…the need for ',
        'I am your ',
        'I see dead ',
        'It’s alive! It’s alive! IT’S ',
        'I ate his liver with some fava beans and a nice ',
        'They may take our lives, but they’ll never take…OUR ',
        'It’s a tr.. ',
        'I’m the king of the ',
        'Pay no attention to that man behind the ',
        'Forget it, Jake. It’s ',
        'Hasta la vista… '
    ]

    if index < 15:
        message = random.choice(phrases) + fruit_list[index] + '\n' \
                  + fruit_description[index][0] + '\n' \
                  + "Energy: " + str(fruit_description[index][1]) + ' kcal ' \
                  + str(fruit_description[index][2]) + ' kJ' + '\n'
        for i, element in enumerate(fruit_description[index][3:]):
            message += fruit_nutrition_list[i] + ': ' + str(element) + ' g\n'
    else:
        message = fruit_list[index] + '\n\n' + str(fruit_description[index])
    return message
