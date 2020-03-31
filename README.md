# Telegram bot for fruit recognition.

This bot will recognize fruit at the photo and help you at local markets all over the world!

## Project parts:

### dataset.py

Images loading in dataset, class labeling.

### training.py

Neuron network train process.

### utils.py

Supporting functions for visualization.

### fruits_general.ipynb

Neuron network training piplene: data loading, setting neuron network architecture, hyperparameters searching, assessment of network accuracy on test data and saving of model weights.

### demo.ipynb

Demonstration of neuron network functions.

### bot.py

Scrypt for bot launcing on a remote server. Predicts a fruit name by photo, responds to text messages by template.
