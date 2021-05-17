"Dataset class for classification model"
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


def prepare_audio(audio_file, resample=22050, n_mels=64):

    sound_tensor, sample_rate = torchaudio.load(os.path.normpath(audio_file),
                                                normalize=True)
    # Resample audio to same format
    if sample_rate != resample:
        resample_transform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample)
        sound_tensor = resample_transform(sound_tensor)

    # This will convert audio files with two channels into one
    sound_tensor = torch.mean(sound_tensor, dim=0, keepdim=True)

    # Convert audio to log-scale Mel spectrogram
    melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=resample, n_mels=n_mels)
    melspectrogram = melspectrogram_transform(sound_tensor)
    melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)

    # Make sure all spectrograms are the same size
    fixed_length = 3 * (resample // 200)
    if melspectogram_db.shape[2] < fixed_length:
        melspectogram_db = torch.nn.functional.pad(melspectogram_db,
                                                   (0, fixed_length - melspectogram_db.shape[2]))
    else:
        melspectogram_db = melspectogram_db[:, :, :fixed_length]
    return sound_tensor, melspectogram_db


class GenocidalClassifierDataset(Dataset):
    DATA_SOURCE = 'D:/ARQUIVOS PESSOAIS/GitHub/GenocidalVoice/data/datasets/'

    def __init__(self, label_0_csv, label_1_csv, resample_freq=22050,
                 n_mels=64, split=1):

        self.n_mels = n_mels
        self.resample = resample_freq
        # Load, shuffle and combine data
        data_0 = pd.read_csv(os.path.relpath(label_0_csv),
                             header=0, dtype=str).reset_index(drop=True)
        data_1 = pd.read_csv(os.path.relpath(label_1_csv),
                             header=0, dtype=str).reset_index(drop=True)

        # Add labels
        data_0["label"] = 0
        data_1["label"] = 1

        # Add source path
        data_0["source"] = (self.DATA_SOURCE +
                            label_0_csv.split("/")[-1].split(".")[0] +
                            "/wav/" + data_0["id"] + ".wav")
        data_1["source"] = (self.DATA_SOURCE +
                            label_1_csv.split("/")[-1].split(".")[0] +
                            "/wav/" + data_1["id"] + ".wav")

        # Keep expected data unbalance 80% positives
        data_0 = data_0.iloc[:int(len(data_1)*.75)]

        # Combine and Split train/test
        index_0 = int(len(data_0) * split)
        index_1 = int(len(data_1) * split)
        if 0 < split <= 1:
            self.data = pd.concat(
                [data_0.iloc[:index_0], data_1.iloc[:index_1]])
        elif -1 <= split < 0:
            self.data = pd.concat(
                [data_0.iloc[index_0:], data_1.iloc[index_1:]])
        else:
            raise Exception("Invalid value of split, [-1 < split < 1]")
        # Remove possible duplicates
        self.data = self.data.drop_duplicates()

    def __getitem__(self, index):
        audio_file = self.data["source"].iloc[index]
        sound_tensor, melspectogram_db = prepare_audio(
            audio_file, self.resample, self.n_mels)

        return melspectogram_db, self.data["label"].iloc[index]

    def __len__(self):
        return len(self.data)


class GenocidalPredictionDataset(Dataset):
    DATA_SOURCE = 'D:/ARQUIVOS PESSOAIS/GitHub/GenocidalVoice/data/datasets/'

    def __init__(self, metadata_csv, resample_freq=22050,
                 n_mels=64, split=1):

        self.n_mels = n_mels
        self.resample = resample_freq
        # Load data
        data = pd.read_csv(os.path.relpath(metadata_csv),
                           header=0, dtype=str).reset_index(drop=True)

        # Add source path
        basename = os.path.basename(os.path.normpath(metadata_csv))
        data["source"] = (self.DATA_SOURCE +
                          basename.split(".")[0] + "/wav/" +
                          data["id"] + ".wav")

        # Remove possible duplicates
        self.data = data.drop_duplicates()

    def __getitem__(self, index):
        audio_file = self.data["source"].iloc[index]
        sound_tensor, melspectogram_db = prepare_audio(
            audio_file, self.resample, self.n_mels)

        return index, self.data["source"].iloc[index], melspectogram_db

    def __len__(self):
        return len(self.data)
