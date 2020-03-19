import numpy as np
import torch
from torch import nn
device = torch.device("cuda:0")

def train_model(model, train_loader, val_loader, 
                loss, optimizer, scheduler, num_epochs):    
    """Функция для тренировки модели.
    Args:
        model (torch.nn): Модель нейронной сети.
        train_loader (torch.utils.data.DataLoader): Dataloader для тренировочных данных.
        val_loader (torch.utils.data.DataLoader): Dataloader для валидационных данных.
        loss (torch.nn): Функция потерь.
        optimizer (torch.optim): Оптимайзер нейронной сети.
        scheduler (torch.optim.lr_scheduler): Sheduler для изменения learning rate сети.
        num_epochs (int): Количество эпох при тренировке.
    """
    loss_history = [] # В этот лист записываются все значения функции потерь в ходе тренировки.
    train_history = [] # В этот лист записываются все значения точности на тренировочных данных в ходе тренировки.
    val_history = [] # В этот лист записываются все значения точности на валидации в ходе тренировки.
    for epoch in range(num_epochs):
        model.train()         
        loss_accum = 0 # Здесь будут складываться суммы всех значений функции потерь на каждом из батчей.
        correct_samples = 0 # Количество верно предсказанных меток класса.
        total_samples = 0 # Суммарное количество предсказанных меток класса.
        
        for i_step, (x, y,_) in enumerate(train_loader):         
            x_gpu = x.to(device) # Отправляем данные на GPU
            y_gpu = y.to(device)
            
            prediction = model(x_gpu)   
            loss_value = loss(prediction, y_gpu) # Значение функции потерь на батче.
            optimizer.zero_grad() # Обнуляем значение градиентов.
            loss_value.backward() # Считаем значение градиентов по функции потерь.
            optimizer.step() # Делам шаг оптимайзера, меняя веса в соответствии с градиентом.
            _, indices = torch.max(prediction, 1) # Находим предсказанные метки классов.
            
            correct_samples += torch.sum(indices==y_gpu) # Находим верно предсказанные метки классов.
            total_samples += y.shape[0]
            loss_accum += loss_value # Добавляем общее количество меток класса в батче.
            
        scheduler.step()
        
        ave_loss = loss_accum / i_step # Считаем среднее значение функции потерь на тренировочных данных в эпохе.
        train_accuracy = float(correct_samples) / total_samples # Cчитаем точность на тренировке.
        val_accuracy = compute_accuracy(model, val_loader) # Cчитаем точность на валидации.
        
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))     
    
    return loss_history, train_history, val_history

def evaluate_model(model, dataset, indices):
    """Функция для получения пары предсказанная метка класса + реальная метка класса.
    Args:
        model (torch.nn): Модель нейронной сети.
        dataset (torch.utils.data.Dataset): Датасет.
        indices (list): Список индексов.
    """
    model.eval()
    val_sampler = SubsetSampler(indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=10, 
                                             sampler=val_sampler)
    predictions = []
    ground_truth = []
    
    for k, (inputs, gt, id) in enumerate(val_loader):
        inputs_gpu = inputs.to(device)
        ground_truth += gt.tolist()
        prediction = model(inputs_gpu)
        _, prediction_semi = torch.max(prediction, 1)
        predictions += prediction_semi.tolist()
    
    return predictions, ground_truth

def compute_accuracy(model, loader):
    """Функция для подсчета точности модели.
    Args:
        model (torch.nn): Модель нейронной сети.
        loader (torch.utils.data.DataLoader): Dataloader для тренировочных данных.
    """
    model.eval() 
    correct_samples = 0
    all_samples = 0
    
    for k, (inputs,classes,id) in enumerate(loader):
        inputs_gpu = inputs.to(device)
        classes_gpu = classes.to(device)
        prediction = model(inputs_gpu)
        _, indices = torch.max(prediction, 1)
        correct_samples += torch.sum(indices==classes_gpu)
        all_samples += classes_gpu.shape[0]
    
    return float(correct_samples) / all_samples
