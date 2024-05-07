import os
import pickle

import pandas as pd

from who_said_that.speaker_assignment import util_components
from who_said_that.video_list import VideoFile


def get_video_stats(
    ground_truth: pd.DataFrame,
    final_video_output: dict,
    final_audio_output: dict,
):

    ground_truth["video_name"] = ground_truth["file_name"].apply(
        lambda x: os.path.splitext(x)[0]
    )
    ground_truth["word_end_modified"] = ground_truth.apply(
        lambda row: (
            (row["word_end"] + 0.01)
            if row["word_start"] == row["word_end"]
            else row["word_end"]
        ),
        axis=1,
    )
    ground_truth["predicted_speakers_video"] = len(final_video_output.keys())
    ground_truth["predicted_speakers_audio"] = len(final_audio_output.keys())

    return ground_truth


def get_audio_video_mapping(final_video_output, final_audio_output):

    final_audio_video_mapping = util_components.get_mapping(
        final_video_output, final_audio_output
    )

    final_video_audio_mapping = util_components.get_mapping(
        final_audio_output, final_video_output
    )

    return final_audio_video_mapping, final_video_audio_mapping


def assign_speakers(
    final_df,
    final_audio_output,
    final_video_output,
    final_video_audio_mapping,
    final_audio_video_mapping,
):

    final_df["a_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_word_to_speaker_mapping(
            row["word_start"],
            row["word_end_modified"],
            final_audio_output,
        ),
        axis=1,
    )

    final_df["v_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_word_to_speaker_mapping(
            row["word_start"],
            row["word_end_modified"],
            final_video_output,
        ),
        axis=1,
    )

    final_df["av_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_speaker_label(
            row["word_start"],
            row["word_end_modified"],
            final_video_output,
            final_audio_output,
            final_video_audio_mapping,
            final_audio_video_mapping,
        ),
        axis=1,
    )

    return final_df


def get_stats(final_df):

    final_df = util_components.match_output(final_df, "a")
    final_df = util_components.match_output(final_df, "v")
    final_df = util_components.match_output(final_df, "av")

    stats_a = util_components.compute_performance(final_df, "a")
    stats_v = util_components.compute_performance(final_df, "v")
    stats_av = util_components.compute_performance(final_df, "av")

    stats = stats_a.merge(stats_v, on="video_name", how="left").merge(
        stats_av, on="video_name", how="left"
    )

    temp_df = final_df[
        ["video_name", "predicted_speakers_audio", "predicted_speakers_video"]
    ].drop_duplicates()

    stats = stats.merge(temp_df, on="video_name", how="left")

    return final_df, stats
