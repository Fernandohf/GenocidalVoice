import os
import argparse
import pandas as pd
import download_youtube_subtitle.main as dys
from youtube_dl import YoutubeDL

VIDEOS_SOURCE = 'data/bolsoanta.csv'
OUTPUT = 'data/raw/'


def download_files(videos, audio_out, text_out):
    # Download audio and subtitles from videos
    options = {'format': 'bestaudio',
               'outtmpl': audio_out + '%(id)s.%(ext)s',
               'nooverwrites': True, }
    audio_downloader = YoutubeDL(options)
    for _, url in videos.items():
        print(f"Saving video/text from {_}")
        try:
            video_id = url.split("v=")[-1]
            audio_downloader.extract_info(url)
            dys.main(video_id, translation=False, to_json=True,
                     output_file=text_out + video_id + ".json")

        except NameError:
            print("No caption available for this videos. Skipping...")
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", help="CSV with video sources in 'urls' column", default=VIDEOS_SOURCE)
    parser.add_argument(
        "--out", help="Output directory", default=OUTPUT)
    args = parser.parse_args()

    # Garantes that the directories exist
    os.makedirs(args.out + "audio/", exist_ok=True)
    os.makedirs(args.out + "text/", exist_ok=True)

    # Load and combine video sources
    if os.path.isfile(args.source):
        videos = pd.read_csv(args.source, header=0)
    else:
        raise Exception(
            f"No source files found! file {args.source} is missing")

    # Download files
    download_files(videos.urls, args.out + "audio/", args.out + "text/")
