# Telegram bot for fruit recognition.

Often during vacation in tropical counties people meet the problem that there are a lot of different unknown fruits in the local market. Locals are not always able to explain which fruit it is. So this bot will recognize fruit for you!

## Project parts:

### dataset.py

Images loading in dataset, class labeling.

### training.py

Neuron network train process.

### utils.py

Supporting function for image visualization with class name.

### fruits_general.ipynb

Neuron network training piplene: data loading, setting neuron network architecture, hyperparameters searching, assessment of network accuracy on test data and saving of model weights.

### demo.ipynb

Demonstration of neuron network functions.

### bot.py

Scrypt for bot launcing on a remote serverPredicts a fruit class by photo, responds to text messages by template.
