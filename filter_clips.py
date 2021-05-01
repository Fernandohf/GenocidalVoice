"""Clean the previously created dataset with the classification model"""
import argparse
import os
from tqdm import tqdm
import pandas as pd
from models.train import last_checkpoint
import torch
from torch.utils.data import DataLoader
from models.classifier.model import GenocidalClassifier
from models.classifier.dataset import GenocidalPredictionDataset


DATASET_PATH = 'data/datasets/ChihuahuaDoTrump'
THRESH = .6


def valid_dataset(dataset_path, model_path, thresh=.5, min_words=3, device="cpu"):
    """Validate the dataset removing invalid audios

    Args:
        dataset_path (str): Path to the dataset
        model_path (str): Path to trained model
        thresh (float, optional): Threshold used for the model classification
        min_words (int, optional): Minimum number of words for each clip

    Returns:
        metadata: Updated dataframe with the valid column for each entry
    """
    metadata_path = os.path.join(dataset_path,
                                 os.path.basename(os.path.normpath(dataset_path)) + ".csv")
    metadata = pd.read_csv(metadata_path, header=0, dtype=str)
    model = GenocidalClassifier(model_path)
    model.to(device)
    model.eval()
    # Audio files
    dataset = GenocidalPredictionDataset(metadata_path)
    dataloader = DataLoader(dataset, batch_size=256)
    # valids = {}
    pbar = tqdm(dataloader)
    for idx, audio_path, audio_spec in pbar:
        # Filter short clips
        # if float(mediainfo(audio_path)['duration']) < min_duration:
        #     metadata.loc[idx, "valid"] = False
        #     # pbar.set_description(f"Invalid entry {audio}")
        #     continue
        audio_spec.to(device)
        valid, prob = model.is_genocidal(audio_spec, thresh)
        metadata.loc[idx.ravel(), "valid"] = valid.ravel()
        metadata.loc[idx.ravel(), "valid_prob"] = prob.ravel()
        # if valid:
        #     msg = f"Valid ({prob*100:2.1f}%):\t{audio}"
        # else:
        #     msg = f"Invalid ({prob*100:2.1f}%):\t{audio}"
        # print(msg)

    # Flag short transcriptions as invalid
    metadata.loc[metadata["transcription"].str.split().str.len() <
                 min_words, "valid"] = "False"
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Folder with the dataset",
        default=DATASET_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = last_checkpoint("checkpoint/classifier")
    metadata = valid_dataset(args.dataset, model_path, THRESH)
    metadata.to_csv(DATASET_PATH + "/metadata.csv")
