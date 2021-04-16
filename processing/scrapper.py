import os
import pandas as pd
import download_youtube_subtitle.main as dys
from youtube_dl import YoutubeDL

VIDEOS_SOURCES = ['data/bolsonaro.csv', 'data/bolsonaro+.csv']
AUDIO_OUT = 'data/raw/audio/'
TEXT_OUT = 'data/raw/text/'


def load_videos(video_urls):
    if os.path.isfile(video_urls):
        videos_df = pd.read_csv(video_urls, header=0)
    else:
        raise Exception(
            f"No source files found! file {video_urls} is missing")
    return videos_df


def download_files(videos_df, audio_out, text_out):
    # Download audio and subtitles from videos
    options = {'format': 'bestaudio',
               'outtmpl': audio_out + '%(id)s.%(ext)s',
               'nooverwrites': True, }
    audio_downloader = YoutubeDL(options)
    for _, url in videos_df.urls.items():
        try:
            video_id = url.split("v=")[-1]
            audio_downloader.extract_info(url)
            dys.main(video_id, translation=False, to_json=True,
                     output_file=text_out + video_id + ".json")
        except NameError:
            print("No caption available for this videos. Skipping...")
            continue


if __name__ == "__main__":

    # Garantes that the directories exist
    os.makedirs(AUDIO_OUT, exist_ok=True)
    os.makedirs(TEXT_OUT, exist_ok=True)

    # Load and combine video sources
    videos_df = []
    for video in VIDEOS_SOURCES:
        videos_df.append(load_videos(video))
    videos = pd.concat(videos_df).drop_duplicates()

    # Download files
    download_files(videos, AUDIO_OUT, TEXT_OUT)
