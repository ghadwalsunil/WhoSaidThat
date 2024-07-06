import os
import pickle
import sys
import time
from typing import List, Literal

import numpy as np
import pandas as pd
from pyannote.metrics.diarization import DiarizationErrorRate
from pydub import AudioSegment

from who_said_that import params
from who_said_that.classes import DiarizationOutput
from who_said_that.evaluation import utils
from who_said_that.utils.components import create_annotation_plot
from who_said_that.video_list import VideoFile


class DiarizationEvaluateResults:
    def __init__(self, video_file: VideoFile, video_diarization_output):

        self.video_file = video_file
        self.video_diarization_output = video_diarization_output

    def perform_evaluation(self, videoDuration):

        sys.stderr.write(
            time.strftime("%Y-%m-%d %H:%M:%S") + " Evaluation of video %s \r\n" % (self.video_file.save_name)
        )

        video_results = {"video_name": self.video_file.save_name}

        if self.video_file.ground_truth_file is None:
            return video_results

        if self.video_file.ground_truth_type == "der":

            if not os.path.isfile(self.video_file.ground_truth_file):
                sys.stderr.write(f"Ground truth file {self.video_file.ground_truth_file} not found\n")
                return None

            metric = DiarizationErrorRate()

            ground_truth = utils.convert_rttm_to_diarization(
                rttm_file=self.video_file.ground_truth_file,
                offset=self.video_file.start,
            )

            create_annotation_plot(
                diarization_output=ground_truth,
                save_path=params.PLOT_OUTPUT_FOLDER,
                video_name=self.video_file.save_name,
                video_duration=videoDuration,
                plot_name="gt",
            )

            pyannote_gt = utils.convert_diarization_output_to_pyannote(ground_truth)

            pyannote_video_simple = utils.convert_diarization_output_to_pyannote(
                self.video_diarization_output["video_simple"]
            )
            video_results["video_simple"] = metric(pyannote_gt, pyannote_video_simple)

            pyannote_audio_12 = utils.convert_diarization_output_to_pyannote(self.video_diarization_output["audio_12"])
            video_results["audio_12"] = metric(pyannote_gt, pyannote_audio_12)

            return video_results

        elif self.video_file.ground_truth_type == "wder":

            final_results = []

            if not os.path.isfile(self.video_file.ground_truth_file):
                sys.stderr.write(f"Ground truth file {self.video_file.ground_truth_file} not found\n")
                return None, None

            ground_truth = pd.read_csv(self.video_file.ground_truth_file, sep="\t")

            video_results["result_type"] = "video_simple"
            _assigned_speakers = utils.compute_word_diarization_error_rate(
                ground_truth, self.video_diarization_output["video_simple"]
            )
            _assigned_speakers.to_excel(
                os.path.join(
                    params.SPEAKER_STATS_FOLDER,
                    f"{self.video_file.save_name}_video_simple.xlsx",
                ),
                index=False,
            )
            video_results.update(_assigned_speakers["matched_result"].value_counts().to_dict())
            final_results.append(video_results)

            video_results = {"video_name": self.video_file.save_name}
            video_results["result_type"] = "audio_12"
            _assigned_speakers = utils.compute_word_diarization_error_rate(
                ground_truth, self.video_diarization_output["audio_12"]
            )
            _assigned_speakers.to_excel(
                os.path.join(
                    params.SPEAKER_STATS_FOLDER,
                    f"{self.video_file.save_name}_audio_12.xlsx",
                ),
                index=False,
            )
            video_results.update(_assigned_speakers["matched_result"].value_counts().to_dict())
            final_results.append(video_results)

            video_results = {"video_name": self.video_file.save_name}
            video_results["result_type"] = "old_simple"
            _assigned_speakers = utils.compute_word_diarization_error_rate_combined(
                ground_truth=ground_truth,
                audio_diarization_output=self.video_diarization_output["audio_12"],
                video_diarization_output=self.video_diarization_output["video_simple"],
            )
            video_results.update(_assigned_speakers["matched_result"].value_counts().to_dict())

            final_results.append(video_results)

            utils.get_additional_info(
                ground_truth=ground_truth,
                audio_diarization_output=self.video_diarization_output["audio_12"],
                video_diarization_output=self.video_diarization_output["video_simple"],
                video_name=self.video_file.save_name,
                save_path=params.SPEAKER_STATS_FOLDER,
            )

            return final_results, _assigned_speakers

        return video_results
