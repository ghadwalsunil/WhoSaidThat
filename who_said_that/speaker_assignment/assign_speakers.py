import os
import pickle

import pandas as pd

from who_said_that.speaker_assignment import utils


class AssignSpeakers:
    def __init__(
        self,
        transcriptions_path: str,
        run_output_path: str,
        video_files: list,
        video_folder: str,
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
        self.transcriptions_path = transcriptions_path
        self.run_output_path = run_output_path
        self.transcriptions_df = pd.read_excel(self.transcriptions_path)
        self.final_video_output = pickle.load(
            open(
                os.path.join(self.run_output_path, "video_diarization_output.pckl"),
                "rb",
            )
        )
        self.final_audio_output = pickle.load(
            open(
                os.path.join(self.run_output_path, "audio_diarization_output.pckl"),
                "rb",
            )
        )

    def perform_speaker_assignment(self):

        final_df = self.transcriptions_df.copy()

        final_df = utils.assign_speakers(
            final_df=final_df,
            final_video_output=self.final_video_output,
            final_audio_output=self.final_audio_output,
            video_files=self.video_files,
        )

        utils.get_stats(
            final_df=final_df,
            final_audio_output=self.final_audio_output,
            final_video_output=self.final_video_output,
            run_output_folder=self.run_output_path,
        )
