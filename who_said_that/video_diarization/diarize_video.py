import os
import pickle
import sys
import time

from pydub import AudioSegment

from who_said_that import params
from who_said_that.utils import components
from who_said_that.video_diarization import utils
from who_said_that.video_list import VideoFile


class VideoDiarization:
    def __init__(self, video_file: VideoFile, video_output_folder):
        self.video_file = video_file
        self.video_output_folder = video_output_folder

    def perform_video_diarization(self, videoDuration):

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S") + " Video diarization of video %s \r\n" % (self.video_file.save_name)
        )
        savePath = os.path.join(self.video_output_folder, self.video_file.save_name)
        pywavPath = os.path.join(savePath, params.PYWAV_FOLDER_NAME)
        pyframesPath = os.path.join(savePath, params.PYFRAMES_FOLDER_NAME)
        pyworkPath = os.path.join(savePath, params.PYWORK_FOLDER_NAME)
        pycropPath = os.path.join(savePath, params.PYCROP_FOLDER_NAME)
        audioFilePath = os.path.join(pywavPath, "audio.wav")

        # Get number of jpg files in pyframesPath
        # try:
        #     numFrames = len([f for f in os.listdir(pyframesPath) if f.endswith(".jpg")])
        # except FileNotFoundError:
        #     numFrames = 7500
        # videoFrameRate = numFrames / videoDuration
        videoFrameRate = 25

        vidTracks = pickle.load(open(os.path.join(pyworkPath, "tracks.pckl"), "rb"))
        scores = pickle.load(open(os.path.join(pyworkPath, "scores.pckl"), "rb"))

        utils.get_track_face_encodings(
            pyworkPath=pyworkPath,
            pyframesPath=pyframesPath,
            tracks=vidTracks,
            scores=scores,
        )

        utils.perform_clustering(pyworkPath=pyworkPath, kmeans_clusters=self.video_file.num_speakers)

        video_output_simple = utils.get_final_tracks(pyworkPath, videoFrameRate, cluster_column="SimpleClusters")

        components.create_annotation_plot(
            diarization_output=video_output_simple,
            save_path=params.PLOT_OUTPUT_FOLDER,
            video_name=self.video_file.save_name,
            video_duration=videoDuration,
            plot_name="video_simple",
        )

        return video_output_simple
