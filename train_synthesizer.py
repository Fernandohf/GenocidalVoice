import os
import torch
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models.synthesizer.text_cleaning import clean_text_series
from models.synthesizer.dataset import VoiceDataset, ResumableRandomSampler
from models.train import train_synthesizer_epoch, load_checkpoint, get_last_checkpoint
from synthesize import save_alignments
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
    # CONFIGURATIONS
    wandb.init(project='genocidal_voice', entity='fernandohf', resume=True)
    config = wandb.config
    config.batch_size = 16
    config.learning_rate = 3.125e-5 * config.batch_size
    config.train_size = 0.85
    config.weight_decay = 1e-6
    config.grad_clip_thresh = 1.0
    config.iters_per_checkpoint = 500
    config.epochs = 100
    config.seed = 1234

    print(
        f"Setting batch size to {config.batch_size}, learning rate to {config.learning_rate}."
    )

    # Set seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Load data
    dataset_dir = args.metadata.rsplit("/", 1)[0]
    print("Loading data...")
    data = pd.read_csv(args.metadata, header=0, dtype=str)
    data["transcription"] = clean_text_series(data.transcription)
    # Filter invalid data
    valid_data = data[data["valid"] == "True"]
    valid_data.loc[:, "source"] = "wav/" + valid_data.loc[:, "id"] + ".wav"
    # Keep same format of old code
    filepaths_and_text = [(s.source, s.transcription)
                          for idx, s in valid_data.iterrows()]
    # Check symbols
    config.symbols = ' abcdefghijklmnopqrstuvwxyzàáâãçéêíóôõúü'
    # symbols = "".join(sorted(set(" ".join(valid_data.transcription))))
    train_cutoff = int(len(filepaths_and_text) * config.train_size)
    train_files = filepaths_and_text[:train_cutoff]
    test_files = filepaths_and_text[train_cutoff:]
    print(f"{len(train_files)} train files, {len(test_files)} test files")
    trainset = VoiceDataset(train_files, dataset_dir,
                            config.symbols, config.seed)
    valset = VoiceDataset(test_files, dataset_dir, config.symbols, config.seed)
    collate_fn = TextMelCollate()
    sampler = ResumableRandomSampler(trainset, config.seed)

    # Load model & optimizer
    print("Loading model...")
    model = Tacotron2().cuda()
    criterion = Tacotron2Loss()
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    last_checkpoint = get_last_checkpoint(args.out)
    # Load last checkpoint
    if last_checkpoint is not None:
        model, optimizer, iteration, initial_epoch, learning_rate, sampler = load_checkpoint(
            last_checkpoint, model, optimizer, sampler)
        iteration += 1
        print(
            f"Loaded checkpoint '{last_checkpoint}' from iteration {iteration}")
    else:
        iteration = 0
        initial_epoch = 0
        learning_rate = config.learning_rate

    # Scheduler
    if initial_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = initial_epoch
    scheduler = StepLR(optimizer, step_size=50, gamma=.5, last_epoch=last_epoch)
    print("Loaded model")

    # Data loaders
    train_loader = DataLoader(
        trainset, num_workers=4, sampler=sampler, batch_size=config.batch_size,
        pin_memory=True, collate_fn=collate_fn, shuffle=False,
    )
    val_loader = DataLoader(
        valset, num_workers=4, sampler=None, batch_size=config.batch_size,
        pin_memory=True, collate_fn=collate_fn, shuffle=False
    )
    print("Loaded data")

    # Training
    model.train()
    for epoch in range(initial_epoch, config.epochs):
        train_synthesizer_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epoch,
            iteration=iteration,
            output_directory=args.out,
            learning_rate=learning_rate,
            grad_clip_thresh=config.grad_clip_thresh,
            iters_per_checkpoint=config.iters_per_checkpoint)

        scheduler.step()
        iteration = 0
        # Saving alignments
        alignment_img = os.path.join(args.out, "alignments", f"epoch_{epoch}.jpg")
        save_alignments(
            model,
            "Boa noite",
            alignment_img
        )
        im = plt.imread(alignment_img)
        wandb.log({"img": [wandb.Image(im, caption="Alignments")]})
        wandb.log({"epoch": epoch})
        print("Epoch: {}".format(epoch))
        print(f"Progress - {epoch}/{config.epochs}")
