import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dataset import Fruits

device = torch.device("cuda:0")

def visualize_samples(dataset, indices, title=None, count=10):
    """Показывает изображения из датасета под указанными индексами.
    Args:
        dataset (torch.utils.data.Dataset): Датасет.
        indices (list): Список индексов.
        title (str): Название для группы изображений (default None).
        count (int): Количество выводимых изображений (default 10).
        
    """
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

def predicting_fruit(model, folder='/demonstration', visualize=True, count=10, 
                     title=None, title_list=None):
    """Функция для демонстрации.
       
       Получает целевую метку для всех изображений в папке и выводит метку вместе с изображением.
       
       Args:
           model (torch.nn): Модель нейронной сети.
           folder (str): Папка с изображениями (default /demonstration).
           visualize (bool): Требуется ли вывод изображений с предсказанными метками (default True).
           count (int): Количество выводимых изображений (default 10).
           title (str): Название для группы изображений (default None).
           title_list (dict): Словарь со значениями классов и их метками (default None).
           
    """ 
    # Датасет используется для вывода изображений.
    orig_dataset = Fruits(folder, production=True) 
    # Датасет используется моделью для получения меток класса.
    load_dataset = Fruits(folder,                  
                          transform=transforms.Compose([
                          transforms.Resize((224, 224)), # Изменяем разрешение на необходимое для работы сети.
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                          ]), production = True  # True, так как исходные данные не содержат данные о метке класа.
                      )
    load_loader = torch.utils.data.DataLoader(load_dataset, batch_size=1)
    
    model.eval()
    predictions = [] # В этот лист будут добавляться метки классов
    
    for k, (inputs, gt, id) in enumerate(load_loader):
        inputs_gpu = inputs.to(device) # Отправляем данные на GPU.
        prediction = model(inputs_gpu) 
        _, prediction_semi = torch.max(prediction, 1)  # Получаем предсказанные метки классов.
        predictions += prediction_semi.tolist()
    
    if visualize == True:  # Выводит изображения, если в функции аргумент True.
        plt.figure(figsize=(count*3,3))
        display_indices = list(range(len(orig_dataset)))
        if title:
            plt.suptitle("%s %s/%s" % (title, len(display_indices), len(orig_dataset)))        
        
        for i, index in enumerate(display_indices):    
            x, _, _ = orig_dataset[index]
            y = title_list[predictions[index]]  # Вместе с изображением выводится предсказанная метка класса.
            plt.subplot(1,count,i+1)
            plt.title("Label: %s" % y)
            plt.imshow(x)
            plt.grid(False)
            plt.axis('off')  
    
    return [title_list[i] for i in predictions]

def imshow(inp, title=None):
    """Функция для вывода изображений из тензора.
    Args:
        inp (torch.tensor): Тензор изображения.
        title (str): Название для группы изображений (default None).
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