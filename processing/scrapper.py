import os
import argparse
import pandas as pd
import download_youtube_subtitle.main as dys
from youtube_dl import YoutubeDL

VIDEOS_SOURCE = 'data/ChihuahuaDoTrump.csv'
OUTPUT = 'data/datasets/'


def download_files(videos, audio_out, text_out):
    # Download audio and subtitles from videos
    options = {'format': 'bestaudio',
               'outtmpl': audio_out + '%(id)s.%(ext)s',
               'nooverwrites': True, }
    audio_downloader = YoutubeDL(options)
    for i, url in videos.items():
        print(f"Saving video/text from {i}")
        try:
            video_id = url.split("v=")[-1]
            audio_downloader.extract_info(url)
            dys.main(video_id, translation=False, to_json=True,
                     output_file=text_out + video_id + ".json")

        except NameError:
            print("No caption available for this videos. Skipping...")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", help="CSV with video sources in 'urls' column", default=VIDEOS_SOURCE)
    parser.add_argument(
        "--out", help="Output directory", default=OUTPUT)
    args = parser.parse_args()

    # Garantes that the directories exist
    dataset_name = args.source.split("/")[-1].split(".")[0]
    text_out = args.out + dataset_name + "/raw/text/"
    audio_out = args.out + dataset_name + "/raw/audio/"
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(text_out, exist_ok=True)

    # Load and combine video sources
    if os.path.isfile(args.source):
        videos = pd.read_csv(args.source, header=0)
    else:
        raise Exception(
            f"No source files found! file {args.source} is missing")

    # Download files
    download_files(videos.urls, audio_out, text_out)
