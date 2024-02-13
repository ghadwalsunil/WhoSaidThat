import os

import pandas as pd
from assign_speakers import util_components


def assign_speakers(
    transcriptions_df: pd.DataFrame,
    final_video_output: dict,
    final_audio_output: dict,
    video_files: list,
):

    transcriptions_df["video_name"] = transcriptions_df["file_name"].apply(
        lambda x: os.path.splitext(x)[0]
    )
    df_videos = pd.DataFrame({"video_name": video_files})
    transcriptions_df = transcriptions_df.merge(df_videos, how="inner")
    transcriptions_df["word_end_modified"] = transcriptions_df.apply(
        lambda row: (
            (row["word_end"] + 0.01)
            if row["word_start"] == row["word_end"]
            else row["word_end"]
        ),
        axis=1,
    )
    transcriptions_df["predicted_speakers_video"] = transcriptions_df[
        "video_name"
    ].apply(lambda x: len(final_video_output[x].keys()))
    transcriptions_df["predicted_speakers_audio"] = transcriptions_df[
        "video_name"
    ].apply(lambda x: len(final_audio_output[x].keys()))

    return transcriptions_df


def get_stats(
    transcription_df, final_audio_output, final_video_output, run_output_folder
):
    final_df = transcription_df.copy()
    final_df["a_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_word_to_speaker_mapping(
            row["word_start"],
            row["word_end_modified"],
            final_audio_output[row["video_name"]],
        ),
        axis=1,
    )

    final_df["v_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_word_to_speaker_mapping(
            row["word_start"],
            row["word_end_modified"],
            final_video_output[row["video_name"]],
        ),
        axis=1,
    )

    final_audio_video_mapping = {}
    final_video_audio_mapping = {}

    for file_name, video_output in final_video_output.items():
        final_audio_video_mapping[file_name] = util_components.get_mapping(
            final_video_output[file_name], final_audio_output[file_name]
        )

    for file_name, video_output in final_video_output.items():
        final_video_audio_mapping[file_name] = util_components.get_mapping(
            final_audio_output[file_name], final_video_output[file_name]
        )

    final_df["av_assigned_speaker"] = final_df.apply(
        lambda row: util_components.get_speaker_label(
            row["video_name"],
            row["word_start"],
            row["word_end_modified"],
            final_video_output,
            final_audio_output,
            final_video_audio_mapping,
            final_audio_video_mapping,
        ),
        axis=1,
    )
    output_file_name = "result_audio_video.xlsx"

    final_df = util_components.match_output(final_df, "a")
    final_df = util_components.match_output(final_df, "v")
    final_df = util_components.match_output(final_df, "av")

    final_df.to_excel(os.path.join(run_output_folder, output_file_name), index=False)

    stats_a = util_components.compute_performance(final_df, "a")
    stats_v = util_components.compute_performance(final_df, "v")
    stats_av = util_components.compute_performance(final_df, "av")

    stats = stats_a.merge(stats_v, on="video_name", how="left").merge(
        stats_av, on="video_name", how="left"
    )

    stats.to_excel(
        os.path.join(run_output_folder, "stats_" + output_file_name), index=False
    )

    return final_df, stats
