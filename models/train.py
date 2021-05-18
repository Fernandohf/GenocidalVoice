""" Functions for traning a model"""
import os
import time
import glob
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import wandb


def load_checkpoint(checkpoint_path, model, optimizer, sampler):
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    sampler.set_state(checkpoint_dict["sampler"])
    iteration = checkpoint_dict["iteration"]
    epoch = checkpoint_dict["epoch"]
    learning_rate = checkpoint_dict["learning_rate"]
    return model, optimizer, iteration, epoch, learning_rate, sampler


def get_last_checkpoint(path):
    last_epoch = 0
    last_iteration = 0
    last_checkpoint = None
    for checkpoint in glob.glob(path + "/*.pt"):
        epoch, iteration = map(int, checkpoint.strip(".pt").split("_")[1:])
        if epoch >= last_epoch:
            last_epoch = epoch
            if iteration > last_iteration:
                last_checkpoint = checkpoint
                last_iteration = iteration
            last_iteration = iteration
            last_checkpoint = checkpoint

    return last_checkpoint


def save_checkpoint(model, optimizer, sampler, learning_rate, iteration, epoch, filepath):
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "epoch": epoch,
            "sampler": sampler.get_state(),
        },
        filepath,
    )


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
    with torch.no_grad():
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
                F1-score: \t {score:.2f}%''')
    return score


def test_synthesizer(model, val_loader, criterion, iteration, console):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)
    model.train()
    console.log("Validation loss at iteration {}: {:9f}  ".format(iteration, val_loss))
    wandb.log({"validation_loss": val_loss})


def train_synthesizer_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epoch,
    output_directory,
    iteration,
    console,
    learning_rate=3.125e-5,
    grad_clip_thresh=1.0,
    iters_per_checkpoint=1000,
):
    model.train()
    for _, batch in enumerate(train_loader):
        start = time.perf_counter()
        model.zero_grad()
        x, y = model.parse_batch(batch)
        y_pred = model(x)

        loss = criterion(y_pred, y)
        reduced_loss = loss.item()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_thresh)
        optimizer.step()

        duration = time.perf_counter() - start
        console.log(
            "[Epoch {:3d}: Iteration {:3d}] - [bold]Train loss[/bold] {:.6f} Grad Norm {:.6f} ({:.2f} s/it)".format(
                epoch, iteration, reduced_loss, grad_norm, duration
            )
        )
        wandb.log({"train_loss": reduced_loss})

        if iteration % iters_per_checkpoint == 0:
            test_synthesizer(model, val_loader, criterion, iteration, console)
            checkpoint_path = os.path.join(
                output_directory, f"checkpoint_{epoch}_{iteration}.pt")
            save_checkpoint(model, optimizer, train_loader.sampler, learning_rate,
                            iteration, epoch, checkpoint_path)
            console.log(
                f"[bold]Checkpoint[/bold] - Epoch: {epoch}, Iteration: {iteration}, File: {checkpoint_path}"
            )

        iteration += 1

    test_synthesizer(model, val_loader, criterion, iteration, console)
    # Reset iterations
    checkpoint_path = os.path.join(
                output_directory, f"checkpointlast_{epoch}_{iteration}.pt")
    save_checkpoint(model, optimizer, train_loader.sampler, learning_rate,
                    iteration, epoch, checkpoint_path)
    iteration = 0
    console.log(
        f"Saving checkpoint at end of epoch {epoch}")
