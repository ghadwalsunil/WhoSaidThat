import os
import pickle
import sys
import time
from typing import List

import pandas as pd

from who_said_that.speaker_assignment import utils
from who_said_that.video_list import VideoFile


class AssignSpeakers:
    def __init__(
        self,
        video_file: VideoFile,
        final_video_output: dict,
        final_audio_output: dict,
    ):
        self.video_file = video_file
        self.final_video_output = final_video_output
        self.final_audio_output = final_audio_output
        self.transcriptions_df = pd.read_csv(
            self.video_file.ground_truth_file, sep="\t"
        )

    def perform_speaker_assignment(self):

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S")
            + " Assigning speakers and getting stats \r\n"
        )

        final_audio_video_mapping, final_video_audio_mapping = (
            utils.get_audio_video_mapping(
                final_video_output=self.final_video_output,
                final_audio_output=self.final_audio_output,
            )
        )

        final_df = utils.get_video_stats(
            ground_truth=self.transcriptions_df,
            final_video_output=self.final_video_output,
            final_audio_output=self.final_audio_output,
        )

        final_df = utils.assign_speakers(
            final_df=final_df,
            final_audio_output=self.final_audio_output,
            final_video_output=self.final_video_output,
            final_video_audio_mapping=final_video_audio_mapping,
            final_audio_video_mapping=final_audio_video_mapping,
        )

        final_df, stats_df = utils.get_stats(
            final_df=final_df,
        )

        final_df.to_excel("../del_later/temp_old.xlsx", index=False)

        return final_df, stats_df
