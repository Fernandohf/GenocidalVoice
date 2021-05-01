import os
import torch
import random
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models.synthesizer.text_cleaning import clean_text_series
from models.train import train_synthesizer_epoch
from models.synthesizer.dataset import VoiceDataset
from tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss


OUTPUT_DIRECTORY = "models/synthesizer/trained"
METADATA_PATH = "data/datasets/ChihuahuaDoTrump/metadata.csv"
# METADATA_PATH = ""


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser()
    parser.add_argument(
        "--out",
        help="Folder to save models and checkpoints",
        default=OUTPUT_DIRECTORY)
    parser.add_argument(
        "--metadata",
        help="CSV file with all metadata info about dataset",
        default=METADATA_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Garantes diretory
    os.makedirs(args.out, exist_ok=True)

    # Hyper-params
    batch_size = 64
    learning_rate = 3.125e-5 * batch_size
    train_size = 0.8
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    iters_per_checkpoint = 1000
    epochs = 8000
    seed = 1234
    # symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    print(
        f"Setting batch size to {batch_size}, learning rate to {learning_rate}."
    )

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # Load model & optimizer
    print("Loading model...")
    model = Tacotron2().cuda()
    optimizer = Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = Tacotron2Loss()
    print("Loaded model")

    # Load data
    dataset_dir = args.metadata.rsplit("/", 1)[0]
    print("Loading data...")
    data = pd.read_csv(args.metadata, header=0, dtype=str)
    data["transcription"] = clean_text_series(data.transcription)
    valid_data = data[data["valid"] == "True"]
    valid_data["source"] = "wav/" + valid_data["id"] + ".wav"
    # Keep same format of old code
    filepaths_and_text = [(s.source, s.transcription)
                          for idx, s in valid_data.iterrows()]
    # Filter invalid data

    # with open(metadata_path, encoding="utf-8") as f:
    #     filepaths_and_text = [line.strip().split("|") for line in f]
    symbols = "".join(sorted(set(" ".join(valid_data.transcription))))

    random.shuffle(filepaths_and_text)
    train_cutoff = int(len(filepaths_and_text) * train_size)
    train_files = filepaths_and_text[:train_cutoff]
    test_files = filepaths_and_text[train_cutoff:]
    print(f"{len(train_files)} train files, {len(test_files)} test files")

    trainset = VoiceDataset(train_files, dataset_dir, symbols, seed)
    valset = VoiceDataset(test_files, dataset_dir, symbols, seed)
    collate_fn = TextMelCollate()

    # Data loaders
    train_loader = DataLoader(
        trainset, num_workers=1, sampler=None, batch_size=batch_size,
        pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valset, num_workers=1, sampler=None, batch_size=batch_size,
        pin_memory=True, collate_fn=collate_fn
    )
    print("Loaded data")

    # Training
    # Load checkpoint if one exists
    iteration = 0

    model.train()
    for epoch in range(1, epochs):
        train_synthesizer_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epoch,
            output_directory=args.out,
            learning_rate=3.125e-5,
            grad_clip_thresh=1.0,
            checkpoint_path=None,
            batch_size=16,
            iters_per_checkpoint=1000,)

        print("Epoch: {}".format(epoch))
        print(f"Progress - {epoch}/{epochs}")
