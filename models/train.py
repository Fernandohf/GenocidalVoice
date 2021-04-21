""" Functions for traning a model"""
import os
import glob
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

CHECKPOINT_PATH = 'checkpoint'


def last_checkpoint(path):
    checkpoints = {}
    for checkpoint in glob.glob(path + "/*"):
        score = float(checkpoints.split("_")[-1].strip(".pt"))
        checkpoints[checkpoint] = score
    return max(checkpoint, key=checkpoints.get)


def train_epoch(model, epoch, criterion, optimizer, dataloader, device, print_very=10):
    """Trains the model for 1 epoch

    Args:
        model: Pytorch model being trained
        criterion: Loss function
        optimizer: Optimizer used in the training
        dataloader: Dataloader for training data
        device: Where the model is being trained
        epoch (int): Current epoch being trained
        print_every (int): Number of batches to wait before printing stats

    """
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        target = torch.unsqueeze(target.to(device), 1).to(torch.float32)
        output = model(data)
        loss = criterion(output, target)
        # Backprogate loss with gradients
        loss.backward()
        optimizer.step()
        # Print stats
        if batch_idx % print_very == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader.dataset), loss))
    return loss.cpu()


def test_binary_classification(model, dataloader, device):
    """Test the model for a binary classification task

    Args:
        model: Pytorch model being tested
        dataloader: Dataloader for testing data
        device: Where the model is being trained
    """
    model.eval()
    model.to(device)
    predictions = []
    labels = []
    for (data, target) in dataloader:
        data = data.to(device)
        target = torch.unsqueeze(target.to(device), 1).to(torch.float32)
        output = torch.sigmoid(model(data))
        pred = (output > .5)
        predictions.append(torch.squeeze(pred).cpu().numpy().astype(int))
        labels.append(torch.squeeze(target).cpu().numpy())
    # Show stats
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    score = f1_score(labels, predictions)
    print(f'''\nTestset metrics ({len(dataloader.dataset)} samples)\n
             Accuracy: \t {accuracy_score(labels, predictions)*100:.2f}%\n
             F1-score: \t {score:.2f}''')
    return score
