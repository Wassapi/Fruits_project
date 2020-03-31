import torch

device = torch.device("cuda:0")


def train_model(model, train_loader, val_loader,
                loss, optimizer, scheduler, num_epochs):
    """Trains model.
    Args:
        model (torch.nn): Model of neuron network.
        train_loader (torch.utils.data.DataLoader): Dataloader for train data.
        val_loader (torch.utils.data.DataLoader): Dataloader for validation data.
        loss (torch.nn): Loss function.
        optimizer (torch.optim): Optimizer for neuron network.
        scheduler (torch.optim.lr_scheduler): Scheduler for learning rate changing.
        num_epochs (int): Number of epochs.
    """
    # Lists for loss and accuracy history
    loss_history = []
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train()

        loss_sum = 0  # Sum of all loss at every batch.
        correct_samples = 0  # Correct samples sum.
        total_samples = 0  # Total samples sum.

        for i_step, (x, y, _) in enumerate(train_loader):
            x_gpu = x.to(device)  # Send to GPU.
            y_gpu = y.to(device)

            prediction = model(x_gpu)
            loss_value = loss(prediction, y_gpu)  # Loss at the batch.
            optimizer.zero_grad()
            loss_value.backward()  # Calculates gradients.
            optimizer.step()  # Changes weights depending of the learning rate ratio.
            _, indices = torch.max(prediction, 1)  # Gets predicted class labels.

            correct_samples += torch.sum(indices == y_gpu)  # Sums number of correct predictions.
            total_samples += y.shape[0]  # Sums samples.
            loss_sum += loss_value  # Sum loss.

        scheduler.step()

        ave_loss = loss_sum / i_step  # Calculate average loss during epoch.
        train_accuracy = float(correct_samples) / total_samples  # Calculate accuracy at train data.
        val_accuracy = compute_accuracy(model, val_loader)  # Calculate accuracy at validation data.

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

    return loss_history, train_history, val_history


def compute_accuracy(model, loader):
    """Returns accuracy of predictions.
    Args:
        model (torch.nn): Model of neuron network.
        loader (torch.utils.data.DataLoader): Dataloader for images.
    """
    model.eval()
    correct_samples = 0
    all_samples = 0

    for inputs, classes, _ in enumerate(loader):
        inputs_gpu = inputs.to(device)
        classes_gpu = classes.to(device)
        prediction = model(inputs_gpu)
        _, indices = torch.max(prediction, 1)
        correct_samples += torch.sum(indices == classes_gpu)
        all_samples += classes_gpu.shape[0]

    return float(correct_samples) / all_samples
