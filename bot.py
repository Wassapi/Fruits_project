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

photo_folder = 'demonstration'
load_folder = os.getcwd() +  '/' + photo_folder   # Folder for image loading.
fruit_list = {0:'Apple', 1:'Banana', 2:'Carambola', 3:'Guava',
              4:'Kiwi', 5:'Mango', 6:'Muskmelon', 7:'Orange', 8:'Peach',
              9:'Pear', 10:'Persimmon', 11:'Pitaya', 12:'Plumу',
              13:'Pomegranet', 14:'Tomato', 15:'Not a fruit'}  # Dict with class names and indices.
classes_nuber = len(fruit_list)
TOKEN = 'YOUR_TOKEN'   # API Telegram token.
bot = telebot.TeleBot(TOKEN)

if os.path.exists(photo_folder) == False:
    os.makedirs(photo_folder)


# Loading of saved model weights.
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, classes_nuber)
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'fruits_project/model_resnet18_comp.sh'), map_location=torch.device('cpu')))


# Functions for telegram bot
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Responds to requests  /start и /help, indroduction to functionality."""
    text = ('Hello, I am fruit bot and I can recognize photo with fruits.\n\n' 
            + 'I already know following fruits: ' 
            + (', '.join(list(fruit_list.values())[:-1])) 
            + '.\n\nSend me the picture of the fruit.\n\n')
    chat_id = message.from_user.id
    bot.send_message(chat_id, text)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    """Responds text messages by requesting of the photo."""
    bot.reply_to(message, "Send me the picture of the fruit.")

@bot.message_handler(content_types=['photo'])
def predict_fruit(photo):
    """Return name of the fruit class as reply to the picture sent by user."""
    load_folder = os.getcwd() + '/' + photo_folder # Folder for image loading.
    try:
        file_id = photo.json['photo'][1]['file_id'] # Get ID of the picture for loading.
    except IndexError:
        file_id = photo.json['photo'][0]['file_id']
    file_info = bot.get_file(file_id)
    file = requests.get(
        'https://api.telegram.org/file/bot{0}/{1}'.format(TOKEN, 
                                                          file_info.file_path)
    )
    out = open(load_folder + '/img.jpg', "wb")
    out.write(file.content)
    out.close() # Write image to the file in the folder.
    
    orig_dataset = Fruits(load_folder, production = True)   # Create dataset with loaded images.
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
        _, prediction_semi = torch.max(prediction, 1)  # Get class labels.
        predictions += prediction_semi.tolist()
    bot.reply_to(photo, [fruit_list[i] for i in predictions])

bot.polling(none_stop=True) # Check new messages.
