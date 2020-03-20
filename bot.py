import numpy as np
import telebot
import datetime
import pytz
import json
import traceback
import requests
import os
from PIL import Image
import torch
from torchvision import models, utils
from torch.utils.data import Dataset, SubsetRandomSampler, Sampler
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import Fruits

photo_folfer = 'demonstration'
os. makedirs(photo_folder)
load_folder = os.getcwd() +  '/' + photo_folfer   # Папка для загрузки изображений.
fruit_list = {0:'Apple', 1:'Banana', 2:'Carambola', 3:'Guava',
              4:'Kiwi', 5:'Mango', 6:'Muskmelon', 7:'Orange', 8:'Peach',
              9:'Pear', 10:'Persimmon', 11:'Pitaya', 12:'Plumу',
              13:'Pomegranet', 14:'Tomato', 15:'Not a fruit'}  # Словарь с классами фруктов.
TOKEN = 'YOUR_TOKEN'   # API token Телеграма.
bot = telebot.TeleBot(TOKEN)
language = 'ru'


# Задаем модель, загружаем к ней веса.
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 16)
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'fruits_project/model_resnet18_comp.sh'), map_location=torch.device('cpu')))


# Определяем функции для телеграм бота
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Отвечает на  запросы /start и /help, рассказывая о функционале бота."""
    text = ('Привет я бот, который знает какой фрукт изображен на фото.\n\n
            Я уже умею различать следующие фрукты: ' 
            + (', '.join(list(fruit_list.values())[:-1])) 
            + '\n\nДля начала работы вышлите фото.\n\n')
    chat_id = message.from_user.id
    bot.send_message(chat_id, text)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    """Обрабатывает все текстовые сообщения, отвечая запросом фотографии фрукта."""
    bot.reply_to(message, "Высылайте фото фрукта, чтобы я смог его распознать")

@bot.message_handler(content_types=['photo'])
def predict_fruit(photo):
    """Функция возвращает предсказанный класс фрукта как ответ на сообщение пользователя с фотографией."""
    load_folder = os.getcwd() + '/' + photo_folder # Определяем папку, куда будет загружено изображение.
    try:
        file_id = photo.json['photo'][1]['file_id'] # Получем id для загрузки изображения.
    except IndexError:
        file_id = photo.json['photo'][0]['file_id']
    file_info = bot.get_file(file_id)
    file = requests.get(
        'https://api.telegram.org/file/bot{0}/{1}'.format(TOKEN, 
                                                          file_info.file_path)
    )
    out = open(load_folder + '/img.jpg', "wb")
    out.write(file.content)
    out.close() # Записываем изображение в файл, который затем прогоняется через модель.
    
    orig_dataset = Fruits(load_folder, production = True)   # Создаем датасеты с загруженным изображением.
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
        prediction = model(inputs)
        _, prediction_semi = torch.max(prediction, 1)  # Получем метки класса для загруженного изображения.
        predictions += prediction_semi.tolist()
    bot.reply_to(photo, [fruit_list[i] for i in predictions])

bot.polling(none_stop=True) # Постоянно проверяем наличие новых сообщений.
