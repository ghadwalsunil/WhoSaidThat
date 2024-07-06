import os
import pickle
import sys
import time
from typing import List

from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from who_said_that import params
from who_said_that.evaluation import utils
from who_said_that.utils import components
from who_said_that.video_list import VideoFile


class AudioDiarization:
    def __init__(
        self,
        video_file: VideoFile,
        video_output_folder,
    ):

        self.video_file = video_file
        self.video_output_folder = video_output_folder

    def perform_audio_diarization(
        self, pretrained_pipeline: Pipeline, pipeline_name, videoDuration
    ):

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Audio Diarization of video %s %s \r\n"
            % (self.video_file.save_name, pipeline_name)
        )
        savePath = os.path.join(self.video_output_folder, self.video_file.save_name)
        pywavPath = os.path.join(savePath, params.PYWAV_FOLDER_NAME)
        audioFilePath = os.path.join(pywavPath, "audio.wav")

        if self.video_file.num_speakers > 0:
            diarization = pretrained_pipeline(
                audioFilePath, num_speakers=self.video_file.num_speakers
            )
        else:
            diarization = pretrained_pipeline(audioFilePath)

        audio_output = utils.convert_pyannote_to_diarization(diarization)

        components.create_annotation_plot(
            diarization_output=audio_output,
            save_path=params.PLOT_OUTPUT_FOLDER,
            video_name=self.video_file.save_name,
            plot_name=f"audio_{pipeline_name}",
            video_duration=videoDuration,
        )

        return audio_output
