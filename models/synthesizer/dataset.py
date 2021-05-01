"Dataset class for classification model"
import os
import torch
import torchaudio
import numpy as np
from scipy.io.wavfile import read
from torch.utils.data import Dataset
from tacotron2_model import TacotronSTFT


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    sound_tensor, sampling_rate = torchaudio.load(os.path.normpath(full_path),
                                                  normalize=True)
    return sound_tensor, sampling_rate


def prepare_audio(audio_file, resample=22050, n_mels=64):

    sound_tensor, sample_rate = torchaudio.load(os.path.normpath(audio_file),
                                                normalize=True)
    # Resample audio to same format
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


class VoiceDataset(Dataset):
    """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, filepaths_and_text, dataset_path, symbols, seed, sample_rate=22050):
        self.filepaths_and_text = filepaths_and_text
        self.dataset_path = dataset_path
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        # self.max_wav_value = 32768.0
        self.sampling_rate = sample_rate
        self.load_mel_from_disk = False
        self.stft = TacotronSTFT()

    def get_mel_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        filepath = os.path.join(self.dataset_path, filename)

        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filepath)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio  # / self.max_wav_value
            #audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(
                audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filepath))
            assert melspec.size(0) == self.stft.n_mel_channels, "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )

        return melspec

    def get_text(self, text):
        sequence = [self.symbol_to_id[s] for s in text if s != "_"]
        text_norm = torch.IntTensor(sequence)
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.filepaths_and_text[index])

    def __len__(self):
        return len(self.filepaths_and_text)
