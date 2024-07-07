import os
import subprocess
import sys
import time


def extract_video(
    input_video_path: str, videoFilePath: str, nDataLoaderThread: int = 10, start_time: float = -1, duration: int = -1
) -> None:
    if start_time == -1 or duration == -1:
        command = "ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % (
            input_video_path,
            nDataLoaderThread,
            videoFilePath,
        )
    else:
        command = "ffmpeg -y -ss %s -i %s -t %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % (
            start_time,
            input_video_path,
            duration,
            nDataLoaderThread,
            videoFilePath,
        )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" % (videoFilePath))


def extract_audio(input_video_path: str, audioFilePath: str, nDataLoaderThread: int = 10) -> None:

    command = "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % (
        input_video_path,
        nDataLoaderThread,
        audioFilePath,
    )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" % (audioFilePath))


def extract_frames(input_video_path: str, pyframesPath: str, nDataLoaderThread: int = 10) -> None:
    command = "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % (
        input_video_path,
        nDataLoaderThread,
        os.path.join(pyframesPath, "%06d.jpg"),
    )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" % (pyframesPath))


# extract_video("StarTalk_Cosmic_780_850.mp4", "video.avi")
    
# file_list = [f for f in os.listdir("../AVA-AVD/dataset/videos") if f.endswith('.mp4')]

# for file in file_list:
#     print(file)
#     extract_audio("../AVA-AVD/dataset/videos/" + file, "waves/" + file[:-4] + ".wav")

# extract_frames("video.avi", "frames")
    
video_path = "../WhoSaidThat/output/video_temp/zzz_PiersMorgan_1_0_165/pycrop/00036.avi"
output_path = "frames"

extract_frames(video_path, output_path)
