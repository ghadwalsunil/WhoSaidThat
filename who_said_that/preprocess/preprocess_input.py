import os
from shutil import rmtree
from typing import List
import sys
import time

from who_said_that import params
from who_said_that.video_list import VideoFile
from who_said_that.preprocess import utils


class Preprocess:
    def __init__(
        self,
        video_file: VideoFile,
        video_folder: str,
        run_output_folder: str,
        video_output_folder: str,
    ):
        self.video_file = video_file
        self.video_folder = video_folder
        self.run_output_folder = run_output_folder
        self.video_output_folder = video_output_folder

    def perform_preprocessing(self) -> bool:

        if not os.path.exists(self.video_output_folder):
            os.mkdir(self.video_output_folder)

        input_video_path = os.path.join(self.video_folder, self.video_file.name + ".mp4")
        if not os.path.isfile(input_video_path):
            input_video_path = os.path.join(self.video_folder, self.video_file.name + ".mkv")
            if not os.path.isfile(input_video_path):
                sys.stderr.write("Video file not found: %s\n" % input_video_path)
                return False

        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Preprocessing video %s \r\n" % (self.video_file.save_name))
        savePath = os.path.join(self.video_output_folder, self.video_file.save_name)
        pyaviPath = os.path.join(savePath, params.PYAVI_FOLDER_NAME)
        pywavPath = os.path.join(savePath, params.PYWAV_FOLDER_NAME)
        pyframesPath = os.path.join(savePath, params.PYFRAMES_FOLDER_NAME)
        pyworkPath = os.path.join(savePath, params.PYWORK_FOLDER_NAME)
        pycropPath = os.path.join(savePath, params.PYCROP_FOLDER_NAME)
        if os.path.exists(savePath):
            rmtree(savePath)
        os.makedirs(pyaviPath, exist_ok=True)
        os.makedirs(pyframesPath, exist_ok=True)
        os.makedirs(pyworkPath, exist_ok=True)
        os.makedirs(pycropPath, exist_ok=True)
        os.makedirs(pywavPath, exist_ok=True)

        # Extract video
        videoFilePath = os.path.join(pyaviPath, "video.mp4")
        utils.extract_video(
            input_video_path=input_video_path,
            videoFilePath=videoFilePath,
            start_time=self.video_file.start,
            duration=self.video_file.duration,
        )

        # Extract audio
        audioFilePath = os.path.join(pywavPath, "audio.wav")
        utils.extract_audio(input_video_path=videoFilePath, audioFilePath=audioFilePath)

        # Extract the video frames
        utils.extract_frames(
            input_video_path=videoFilePath,
            pyframesPath=pyframesPath,
        )

        return True
