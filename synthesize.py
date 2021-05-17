import argparse
import os
import IPython.display as ipd
from tacotron2_model import Tacotron2
import torch
import numpy as np
from models.synthesizer.text_cleaning import clean_text
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


SYMBOLS = ' abcdefghijklmnopqrstuvwxyzàáâãçéêíóôõúü'
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
WAVEGLOW_PATH = 'models/waveglow/waveglow_256channels_universal_v5.pt'
MODEL_PATH = 'models/synthesizer/trained/checkpoint_5000'
TEXT = "boa noite"
AUDIO_PATH = "results/example1.wav"
GRAPH_PATH = "results/example1.jpg"


def load_model(model_path):
    if torch.cuda.is_available():
        model = Tacotron2().cuda()
        model.load_state_dict(torch.load(model_path)["state_dict"])
        _ = model.cuda().eval().half()
    else:
        model = Tacotron2()
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device("cpu"))["state_dict"])
    return model


def load_waveglow(waveglow_path):
    waveglow = torch.load(waveglow_path)["model"]
    if torch.cuda.is_available():
        waveglow.cuda().eval().half()

    for k in waveglow.convinv:
        k.float()
    return waveglow


def generate_graph(alignments, filepath):
    data = alignments.float().data.cpu().numpy()[0].T
    plt.imshow(data, aspect="auto", origin="lower", interpolation="none")
    plt.savefig(filepath)

    # plt.imsave(filepath, data, origin="lower", interpolation="none")


def generate_audio(mel, waveglow, filepath, sample_rate=22050):
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=0.666)

    audio = audio[0].data.cpu().numpy()
    audio = ipd.Audio(audio, rate=sample_rate)
    with open(filepath, "wb") as f:
        f.write(audio.data)


def text_to_sequence(text):
    sequence = np.array([[SYMBOL_TO_ID[s] for s in text if s in SYMBOL_TO_ID]])
    if torch.cuda.is_available():
        return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    else:
        return torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()


def save_alignments(model, text, filepath):
    text = clean_text(text)
    sequence = text_to_sequence(text)
    _, _, _, alignments = model.inference(sequence)

    generate_graph(alignments, filepath)


def synthesize(model, waveglow_model, text, graph=None, audio=None):
    text = clean_text(text)
    sequence = text_to_sequence(text)
    _, mel_outputs_postnet, _, alignments = model.inference(sequence)

    if graph:
        generate_graph(alignments, graph)

    if audio:
        generate_audio(mel_outputs_postnet, waveglow_model, audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str,
                        help="tacotron2 model path",
                        default=MODEL_PATH)
    parser.add_argument("-w", "--waveglow_path",
                        type=str, help="waveglow model path",
                        default=WAVEGLOW_PATH)
    parser.add_argument("-t", "--text", type=str,
                        help="text to synthesize",
                        default=TEXT)
    parser.add_argument("-g", "--graph_output_path", type=str,
                        help="path to save alignment graph to",
                        default=GRAPH_PATH)
    parser.add_argument("-a", "--audio_output_path", type=str,
                        help="path to save output audio to", default=AUDIO_PATH)
    args = parser.parse_args()

    assert os.path.isfile(args.model_path), "Model not found"
    assert os.path.isfile(args.waveglow_path), "Waveglow model not found"

    model = load_model(args.model_path)
    waveglow_model = load_waveglow(args.waveglow_path)

    synthesize(
        model,
        waveglow_model,
        args.text,
        args.graph_output_path,
        args.audio_output_path,
    )
