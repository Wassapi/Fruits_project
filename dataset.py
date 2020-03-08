import numpy as np
import pandas as pd
from torch.utils.data import Dataset, SubsetRandomSampler, SequentialSampler, Sampler
import os
from PIL import Image

#Creating our own dataset
class Fruits(Dataset):
    def __init__(self, folder, transform = None, production = False):
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
        return len(self.listname)
    
    def __getitem__(self, index):        
        img_id = None
        img_id = self.listname[index]
        img = Image.open(img_id).convert('RGB')
        if self.production == True:
            y = 77
        else:       
            if 'Apple' in img_id: 
                y = 0
            elif 'Banana' in img_id:
                y = 1
            elif 'Carambola' in img_id:
                y = 2
            elif 'Guava' in img_id:
                y = 3
            elif 'Kiwi' in img_id:
                y = 4
            elif 'Mango' in img_id:
                y = 5
            elif 'Muskmelon' in img_id:
                y = 6
            elif 'Orange' in img_id:
                y = 7
            elif 'Peach' in img_id:
                y = 8
            elif 'Pear' in img_id:
                y = 9
            elif 'Persimmon' in img_id:
                y = 10
            elif 'Pitaya' in img_id:
                y = 11
            elif 'Plumу' in img_id :
                y = 12
            elif 'Pomegranet' in img_id:
                y = 13
            elif 'Tomato' in img_id:
                y = 14
            else:
                y = 15
        
        if self.transform:
            img = self.transform(img)
        return img, y, img_id

class SubsetSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)