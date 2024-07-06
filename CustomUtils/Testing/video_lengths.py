import os
import pandas as pd
from pydub import AudioSegment


def get_audio_length(audio_file):
    audio = AudioSegment.from_file(audio_file)
    return len(audio) / 1000


def main():
    VIDEO_FOLDER = "/vol3/sunil/FinalDataset"
    audio_dir = "/home/sunil/projects/Stuff/Combined/WhoSaidThat/output/video_temp/"
    video_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(VIDEO_FOLDER, "videos"))
        if os.path.isfile(os.path.join(VIDEO_FOLDER, "videos", f)) and os.path.splitext(f)[1] in [".mp4", ".mkv"]
    ]

    video_lengths = []
    for video_name in video_names:
        audio_file = os.path.join(audio_dir, video_name, "pywav", "audio.wav")
        audio_length = get_audio_length(audio_file)
        video_lengths.append({"video_name": video_name, "audio_length": audio_length})

    df = pd.DataFrame(video_lengths)
    df.to_excel("audio_lengths.xlsx", index=False)


if __name__ == "__main__":

    main()
