""" Script for traning the classification model"""
import argparse
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.classifier.model import GenocidalClassifier
from models.classifier.dataset import GenocidalClassifierDataset
from models.train import train_epoch, test_binary_classification, last_checkpoint

EPOCHS = 12
BATCH_SIZE = 256


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        help="Number of epochs to train for",
        type=int,
        default=EPOCHS)
    parser.add_argument(
        "--batchsize",
        help="Size of batch size limited by RAM",
        type=int,
        default=BATCH_SIZE)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    best_model = last_checkpoint('checkpoint/classifier')
    print(f"Loading model {best_model}")
    model = GenocidalClassifier(best_model)
    model.to(device)
    criteria = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.0001)

    # Datasets
    train_set = GenocidalClassifierDataset('data/datasets/outros/outros.csv',
                                           'data/datasets/bolsoanta/bolsoanta.csv',
                                           resample_freq=22050, n_mels=64, split=.8)
    test_set = GenocidalClassifierDataset('data/datasets/outros/outros.csv',
                                          'data/datasets/bolsoanta/bolsoanta.csv',
                                          resample_freq=22050, n_mels=64, split=-.2)
    kwargs = {'num_workers': 1,
              'pin_memory': True} if device == 'cuda' else {}
    train_loader = DataLoader(
        train_set, batch_size=args.batchsize, shuffle=True, **kwargs)
    test_loader = DataLoader(
        test_set, batch_size=args.batchsize, shuffle=True, **kwargs)

    best_score = 0
    for epoch in range(1, args.epochs):

        loss = train_epoch(model, epoch, criteria,
                           optimizer, train_loader, device)
        f1_score = test_binary_classification(model, test_loader, device)
        # Save checkpoint
        if f1_score > best_score:
            best_score = f1_score
            os.makedirs('checkpoint/classifier', exist_ok=True)
            model.save_model(
                f'checkpoint/classifier/model_checkpoint_score_{f1_score:1.2f}.pt')
