import os
import sys
import time
from typing import List

from who_said_that import params
from who_said_that.models.talkNet import talkNet
from who_said_that.talkNetASD import utils
from who_said_that.video_list import VideoFile


class TalkNetASD:
    def __init__(
        self,
        video_file: VideoFile,
        run_output_folder: str,
        video_output_folder: str,
        talkNetModel: talkNet,
        generate_visualization: bool = False,
        saveMarkedFrames: bool = False,
    ):

        self.video_file = video_file
        self.run_output_folder = run_output_folder
        self.video_output_folder = video_output_folder
        self.takNetModel = talkNetModel
        self.generate_visualization = generate_visualization
        self.saveMarkedFrames = saveMarkedFrames

    def perform_talkNetASD(self):
        # Scene detection for the video frames
        savePath = os.path.join(self.video_output_folder, self.video_file.save_name)
        pyaviPath = os.path.join(savePath, params.PYAVI_FOLDER_NAME)
        pywavPath = os.path.join(savePath, params.PYWAV_FOLDER_NAME)
        pyframesPath = os.path.join(savePath, params.PYFRAMES_FOLDER_NAME)
        pyworkPath = os.path.join(savePath, params.PYWORK_FOLDER_NAME)
        pycropPath = os.path.join(savePath, params.PYCROP_FOLDER_NAME)
        videoFilePath = os.path.join(pyaviPath, "video.mp4")
        audioFilePath = os.path.join(pywavPath, "audio.wav")

        utils.scene_detect(
            videoFilePath=videoFilePath,
            pyworkPath=pyworkPath,
            pyframesPath=pyframesPath,
        )
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Scene detection and save in %s \r\n" % (pyworkPath)
        )

        # Face detection for the video frames
        utils.inference_video(
            videoFilePath=videoFilePath,
            pyframesPath=pyframesPath,
            pyworkPath=pyworkPath,
        )
        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Face detection and save in %s \r\n" % (pyworkPath)
        )

        # Face tracking
        utils.track_faces(
            pyworkPath=pyworkPath,
        )

        # Face clips cropping
        utils.crop_face_clips(
            pyworkPath=pyworkPath,
            pycropPath=pycropPath,
            pyframesPath=pyframesPath,
            audioFilePath=audioFilePath,
        )

        # Active Speaker Detection by TalkNet
        utils.talknet_speaker_detection(
            pycropPath=pycropPath,
            pyworkPath=pyworkPath,
            talkNetModel=self.takNetModel,
        )

        if self.generate_visualization:
            utils.visualization(
                pyframesPath=pyframesPath,
                pyaviPath=pyaviPath,
                pyworkPath=pyworkPath,
                pywavPath=pywavPath,
                saveMarkedFrames=self.saveMarkedFrames,
            )
            sys.stderr.write(
                time.strftime("%Y-%m-%d %H:%M:%S")
                + " Visualization and save in %s \r\n" % (pyaviPath)
            )
