"""Clean the previously created dataset with the classification model"""

import os
import argparse
import pandas as pd
from models.classifier.model import GenocidalClassifier

DATASET_PATH = 'data/datasets/ChihuahuaDoTrump'


def clean_dataset(dataset_path, model_path, ):
    metadata = pd.read_csv(dataset_path + ".csv", header=0)
    model = GenocidalClassifier(model_path)

    model.is_genocidal()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Folder contaning the audio/text files downloaded")
    args = parser.parse_args()

    metadata_path = os.path.join(args.dataset, os.path.basename(os.path.normpath(args.dataset)) + "csv")
