import os

import requests
import telebot
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

from dataset import Fruits, fruit_list, message_with_nutrition

photo_folder = 'demonstration'
load_folder = os.getcwd() + '/' + photo_folder  # Folder for image loading.

classes_number = len(fruit_list)
TOKEN = 'YOUR_TOKEN'  # API Telegram token.
bot = telebot.TeleBot(TOKEN)

if not os.path.exists(photo_folder):
    os.makedirs(photo_folder)

# Loading of saved model weights.
model = models.resnet18(pretrained=False)
num_filters = model.fc.in_features
model.fc = nn.Linear(num_filters, classes_number)
model.load_state_dict(torch.load(
    os.path.join(os.getcwd(), 'fruits_project/model_resnet18_comp.sh'),
    map_location=torch.device('cpu'))
)


# Functions for telegram bot
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Responds to requests  /start и /help, introduction to functionality."""
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
    load_path = os.getcwd() + '/' + photo_folder  # Folder for image loading.
    try:
        file_id = photo.json['photo'][1]['file_id']  # Get ID of the picture for loading.
    except IndexError:
        file_id = photo.json['photo'][0]['file_id']
    file_info = bot.get_file(file_id)
    file = requests.get(
        'https://api.telegram.org/file/bot{0}/{1}'.format(TOKEN,
                                                          file_info.file_path)
    )
    out = open(load_path + '/img.jpg', "wb")
    out.write(file.content)
    out.close()  # Write image to the file in the folder.

    load_dataset = Fruits(load_path,
                          transform=transforms.Compose([
                              transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                          ]), production=True
                          )
    load_loader = torch.utils.data.DataLoader(load_dataset, batch_size=1)
    model.eval()
    predictions = []
    for k, (inputs, _, _) in enumerate(load_loader):
        prediction = model(inputs)
        _, prediction_semi = torch.max(prediction, 1)  # Get class labels.
        predictions += prediction_semi.tolist()
    bot.reply_to(photo,
                 [message_with_nutrition(i) for i in predictions]
                 )


bot.polling(none_stop=True)  # Check new messages.
