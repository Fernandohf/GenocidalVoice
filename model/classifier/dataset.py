"Dataset class for classification model"
import os
import pandas as pd
import torch
from torch import Dataset


class GenocidalClassfierDataset(Dataset):
    def __init__(self, csv_path, audio_path, resample_freq=22050, n_mels=64):
        self.audio_path = audio_path
        self.n_mels = n_mels
        self.resample = resample_freq
        self.data = pd.read_csv(csv_path).sample(frac=1).reset_index(drop=True)
        
    def __getitem__(self, index):
        audio_file = os.path.join(self.audio_path, )
        soundData, sample_rate = torchaudio.load(audio_file, out=None,
            normalization=True)

        if self.resample > 0:
            resample_transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample)
            soundData = resample_transform(soundData)
            
        # This will convert audio files with two channels into one
        soundData = torch.mean(soundData, dim=0, keepdim=True)
                
        # Convert audio to log-scale Mel spectrogram
        melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.resample, n_mels=self.n_mels)
        melspectrogram = melspectrogram_transform(soundData)
        melspectogram_db =
            torchaudio.transforms.AmplitudeToDB()(melspectrogram)
            
        #Make sure all spectrograms are the same size
        fixed_length = 3 * (self.resample//200)
        if melspectogram_db.shape[2] < fixed_length:
            melspectogram_db = torch.nn.functional.pad(melspectogram_db, 
                (0, fixed_length - melspectogram_db.shape[2]))
        else:
            melspectogram_db = melspectogram_db[:, :, :fixed_length]

        return soundData, self.resample, melspectogram_db, self.labels[index]