import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from more_itertools import pairwise
from pydub import AudioSegment
from pydub.effects import normalize

DATASET_NAME = 'genocida'
DATASET_OUT = 'data/datasets/'
DATASET_SOURCE = 'data/bolsoanta.csv'
BIT_RATE = "32k"
# TARGET_SAMPLE_RATE = 22050


def fix_durations(subtitles_list):
    """Fix duration in subtitles json

    Args:
        subtitles_list: List of dictionaries with subtitle info

    Returns:
        [list(dict)]: Fixed list of dictionaries
    """
    for s1, s2 in pairwise(subtitles_list):
        s1['dur'] = str(float(s2['start']) - float(s1['start']))
    return subtitles_list


def split_fragments(text_file, audio_file, data_out, init_uid):
    """Create the pairs phrases/transcription in the dataset format
    from given audio/text files

    Args:
        text_file (str): Path to text file
        audio_file (str): Path to audio file
        data_out (str): Path to dataset output
        init_uid (int): Initial value for unique identifier

    Returns:
        metadata: Dataframe with all info of the dataset
        uid (int): Last used unique identifier
    """
    # Load subtitle
    text = json.load(open(text_file, encoding='utf8'))
    fixed_text = fix_durations(text['original'])

    # Load sound
    sound = AudioSegment.from_file(audio_file)
    fragments = []
    transcriptions = []
    ids = []
    uid = init_uid
    # Splitting by text sentence
    for sentence in fixed_text:
        silence = AudioSegment.silent(duration=100)
        start = int(float(sentence['start']) * 1000) - 300
        end = int(float(sentence['start']) * 1000 +
                  float(sentence['dur']) * 1000)
        # Add 100ms silent around audio
        fragment = normalize(silence + sound[start:end] + silence)
        uid += 1
        str_id = "{:0>6}".format(uid)
        ids.append(str_id)
        fragments.append(fragment)
        transcriptions.append(sentence['text'])
        fragment.export(data_out + "wav/" + str_id + ".wav",
                        format="wav", bitrate=BIT_RATE)

    # Create dataset
    metadata = pd.DataFrame(columns=["id", "transcription"])
    metadata["id"] = ids
    metadata["transcription"] = transcriptions

    return (metadata, uid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Dataset's name", default=DATASET_NAME)
    parser.add_argument(
        "--out", help="Output folder'", default=DATASET_OUT)
    parser.add_argument(
        "--source",
        help="Source file with all files names",
        default=DATASET_SOURCE)
    args = parser.parse_args()

    # Garantes folder exists
    os.makedirs(DATASET_OUT + args.name + "/wav/", exist_ok=True)
    data_out = DATASET_OUT + args.name + "/"
    # List of Dataframes
    dfs = []
    uid = 0
    # Source
    diretory = args.source.split(".")[0] + "/raw/"
    data_source = (diretory + "audio/" +
                   pd.read_csv(args.source).urls.str.split("v=").str[-1] + ".m4a")

    pbar = tqdm(data_source, desc="Processing files...")
    total = 0
    for audio_file in pbar:
        file_name = audio_file.split('/')[-1]
        # Check if subtitle exists
        file_no_ext = file_name.split(".")[0]
        text_file = diretory + "text/" + file_no_ext + ".json"
        if os.path.exists(text_file):
            df, uid = split_fragments(text_file, audio_file, data_out, uid)
            total += df.shape[0]
            pbar.set_description(
                f"{file_name} added {df.shape[0]}/{total} fragments to dataset")
            dfs.append(df)
        # Ignore audio without text
        else:
            pbar.set_description(f"Ignoring {file_name} no subtitles found.")
            continue
    dataset = pd.concat(dfs).reset_index()
    dataset.to_csv(data_out + args.name + ".csv", encoding='utf8')
