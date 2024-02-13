import os
from shutil import rmtree
from typing import List

import params
from preprocess import utils


class Preprocess:
    def __init__(
        self,
        video_files: List[str],
        video_folder: str,
        run_output_folder: str,
        video_output_folder: str,
    ):
        if not video_files:
            self.video_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(video_folder)
                if os.path.isfile(os.path.join(video_folder, f))
                and os.path.splitext(os.path.join(video_folder, f))[1] in [".mp4"]
            ]
        else:
            self.video_files = video_files

        self.video_folder = video_folder
        self.run_output_folder = run_output_folder
        self.video_output_folder = video_output_folder

    def perform_preprocessing(self) -> None:

        if os.path.exists(self.run_output_folder):
            rmtree(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        os.mkdir(params.SRT_OUTPUT_FOLDER)
        os.mkdir(params.JS_OUTPUT_FOLDER)
        os.mkdir(params.PLOT_OUTPUT_FOLDER)

        if os.path.exists(self.video_output_folder):
            rmtree(self.video_output_folder)
        os.mkdir(self.video_output_folder)

        for video_file in self.video_files:
            savePath = os.path.join(self.video_output_folder, video_file)
            pyaviPath = os.path.join(savePath, params.PYAVI_FOLDER_NAME)
            pyframesPath = os.path.join(savePath, params.PYFRAMES_FOLDER_NAME)
            pyworkPath = os.path.join(savePath, params.PYWORK_FOLDER_NAME)
            pycropPath = os.path.join(savePath, params.PYCROP_FOLDER_NAME)
            if os.path.exists(savePath):
                rmtree(savePath)
            os.makedirs(pyaviPath, exist_ok=True)
            os.makedirs(pyframesPath, exist_ok=True)
            os.makedirs(pyworkPath, exist_ok=True)
            os.makedirs(pycropPath, exist_ok=True)

            input_video_path = os.path.join(self.video_folder, video_file + ".mp4")

            # Extract video
            videoFilePath = os.path.join(pyaviPath, "video.avi")
            utils.extract_video(
                input_video_path=input_video_path, videoFilePath=videoFilePath
            )

            # Extract audio
            audioFilePath = os.path.join(pyaviPath, "audio.wav")
            utils.extract_audio(
                input_video_path=videoFilePath, audioFilePath=audioFilePath
            )

            # Extract the video frames
            utils.extract_frames(
                input_video_path=videoFilePath,
                pyframesPath=pyframesPath,
            )
