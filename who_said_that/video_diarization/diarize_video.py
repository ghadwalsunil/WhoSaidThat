import os
import pickle
import sys
import time

from pydub import AudioSegment

from who_said_that import params
from who_said_that.utils import components
from who_said_that.video_diarization import utils


class VideoDiarization:
    def __init__(
        self, video_files, video_folder, run_output_folder, video_output_folder
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

    def perform_video_diarization(self):

        final_video_output = {}
        for video_file in self.video_files:
            sys.stderr.write(
                time.strftime("%Y-%m-%d %H:%M:%S")
                + " Processing video %s \r\n" % (video_file)
            )
            savePath = os.path.join(self.video_output_folder, video_file)
            pyaviPath = os.path.join(savePath, params.PYAVI_FOLDER_NAME)
            pyframesPath = os.path.join(savePath, params.PYFRAMES_FOLDER_NAME)
            pyworkPath = os.path.join(savePath, params.PYWORK_FOLDER_NAME)
            pycropPath = os.path.join(savePath, params.PYCROP_FOLDER_NAME)
            audioFilePath = os.path.join(savePath, "pyavi", "audio.wav")

            # Get number of jpg files in pyframesPath
            numFrames = len([f for f in os.listdir(pyframesPath) if f.endswith(".jpg")])
            videoDuration = len(AudioSegment.from_file(audioFilePath)) / 1000
            videoFrameRate = numFrames / videoDuration

            vidTracks = pickle.load(open(os.path.join(pyworkPath, "tracks.pckl"), "rb"))
            scores = pickle.load(open(os.path.join(pyworkPath, "scores.pckl"), "rb"))

            utils.get_track_face_encodings(
                pyworkPath=pyworkPath,
                pyframesPath=pyframesPath,
                tracks=vidTracks,
                scores=scores,
            )

            video_output = utils.get_final_tracks(pyworkPath, videoFrameRate)

            final_video_output[video_file] = video_output

            components.create_annotation_plot(
                diarization_output=video_output,
                save_path=params.PLOT_OUTPUT_FOLDER,
                video_name=video_file,
                video_duration=videoDuration,
                plot_name="video",
            )

        with open(
            os.path.join(self.run_output_folder, "video_diarization_output.pckl"), "wb"
        ) as f:
            pickle.dump(final_video_output, f)

        return final_video_output
