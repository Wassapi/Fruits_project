import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler, Sampler
import os
from PIL import Image

fruit_list = {0:'Apple', 1:'Banana', 2:'Carambola', 3:'Guava',
              4:'Kiwi', 5:'Mango', 6:'Muskmelon', 7:'Orange', 8:'Peach',
              9:'Pear', 10:'Persimmon', 11:'Pitaya', 12:'Plumу',
              13:'Pomegranet', 14:'Tomato', 15:'Not a fruit'}

class Fruits(Dataset):
    """Создает датасет из всех изображений в папке и её подпапок"""
    def __init__(self, folder, transform=None, production=False):
        """ Создает список всех файлов в папке с датасетом
        Args:
            folder (str): Имя папки в фотографиями для датасета.
            transform (torch.transforms): Преобразования, которые делаются с файлами фотографий (default None).
            production (bool): False работа с размеченными файлами True с неразмеченными (default False).
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
        """Возвращает количество файлов в датасете."""
        return len(self.listname)
    
    def __getitem__(self, index):
        """Возвращает изображение, метку класса и путь до файла изображения по индексу.
        Args:
            index (int): Индекс элемента в датасете.
        """
        img_id = None
        img_id = self.listname[index]
        img = Image.open(img_id).convert('RGB')
        if self.production == True: # True мы используем, когда не знаем реальной метки класса
                                    # (используется в демонстрации и телеграм боте).
            y = 77
        else: # В остальных случаях мы можем получить реальную метку класса из названия изображения.
            inv_fruit_list = dict(zip(fruit_list.values(), fruit_list.keys()))
            class_title = str.split(img_id, sep= '/')[1] # Получаем имя класса как названия папки, в которой хранится значение. 
            if class_title == 'Other':
                y = 15
            else:
                #Получем метку класса по его имени.
                try:
                    y = inv_fruit_list[class_title]
                except KeyError:
                    raise KeyError(
                        'Присутствует папка с неизвестным классом в папке с данными:', 
                        class_title)

        if self.transform:
            img = self.transform(img)
        return img, y, img_id

    
class SubsetSampler(Sampler):
    """Класс для сэмплера, который возвращает индексы по порядку."""

    def __init__(self, indices):
        """Присваивает список индексов.
        Args:
            indices (list): Список индексов.
        """
        self.indices = indices

    def __iter__(self):
        """Итерация по списку индексов."""
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        """Возвращает количество индексов в списке."""
        return len(self.indices)